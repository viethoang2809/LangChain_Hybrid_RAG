# app/utils/vietmap_utils.py
# ===============================================================
# Các hàm gọi VietMap API: lấy toạ độ, địa điểm xung quanh,
# và cập nhật enrichment trực tiếp vào chat_history.jsonl
# ===============================================================

import os
import re
import json
import requests
from dotenv import load_dotenv

# --- Load API key từ .env ---
load_dotenv()
VIETMAP_API_KEY = os.getenv("VIETMAP_API_KEY")
BASE_URL = "https://maps.vietmap.vn/api"

# ===============================================================
# 1️⃣ HÀM CƠ BẢN: GỌI API
# ===============================================================
def _get(url: str, params: dict):
    """Gọi API VietMap với error handling."""
    res = requests.get(url, params=params, timeout=15)
    res.raise_for_status()
    return res.json()


def search_address(address: str):
    """Tìm địa chỉ → lấy ref_id, lat/lng qua Search v4."""
    url = f"{BASE_URL}/search/v4"
    params = {"apikey": VIETMAP_API_KEY, "text": address, "display_type": 1}
    data = _get(url, params)
    if not data:
        return None
    ref_id = data[0].get("ref_id")
    return get_place_detail(ref_id)


def get_place_detail(ref_id: str):
    """Lấy chi tiết toạ độ qua Place v4."""
    url = f"{BASE_URL}/place/v4"
    params = {"apikey": VIETMAP_API_KEY, "refid": ref_id}
    data = _get(url, params)
    return {
        "lat": data.get("lat"),
        "lng": data.get("lng"),
        "city": data.get("city"),
        "ward": data.get("ward"),
        "district": data.get("district", ""),
    }

# ===============================================================
# 2️⃣ TÌM ĐỊA ĐIỂM QUANH TỌA ĐỘ THEO TỪ KHÓA
# ===============================================================
def search_nearby_by_keyword(lat: float, lng: float, keyword: str, max_results: int = 3):
    """Tìm các địa điểm quanh toạ độ bằng keyword (ví dụ: bệnh viện, siêu thị, công viên...)."""
    url = f"{BASE_URL}/search/v4"
    params = {
        "apikey": VIETMAP_API_KEY,
        "text": keyword,
        "focus": f"{lat},{lng}",
        "display_type": 1,
    }
    data = _get(url, params)
    seen, results = set(), []
    for item in sorted(data, key=lambda x: x.get("distance", 9999)):
        name, address = item.get("name", ""), item.get("display", "")
        key = f"{name}-{address}"
        if key not in seen and name:
            results.append({
                "name": name.strip(),
                "address": address.strip(),
                "distance_km": round(item.get("distance", 0), 2),
            })
            seen.add(key)
        if len(results) >= max_results:
            break
    return results


def get_nearby_groups(lat: float, lng: float, max_results: int = 3):
    """
    Gom các nhóm tiện ích quanh 1 toạ độ:
    bệnh viện, trường học (3 cấp), siêu thị, chợ, công viên, rạp phim, nhà hàng, cà phê.
    """
    KEYWORDS = {
        "hospital": "bệnh viện",
        "primary_school": "tiểu học",
        "secondary_school": "trung học",
        "university": "đại học",
        "supermarket": "siêu thị",
        "park": "công viên",
        "restaurant": "nhà hàng",
    }

    grouped = {}
    for group, keyword in KEYWORDS.items():
        try:
            grouped[group] = search_nearby_by_keyword(lat, lng, keyword, max_results)
        except Exception as e:
            grouped[group] = {"error": str(e)}
    return grouped


def get_nearby_places(lat: float, lng: float, radius_m: int = 1000, limit: int = 20):
    """(Legacy) Lấy danh sách POI xung quanh theo bán kính."""
    url = f"{BASE_URL}/search/v4"
    params = {
        "apikey": VIETMAP_API_KEY,
        "circle_center": f"{lat},{lng}",
        "circle_radius": radius_m,
        "layers": "POI",
        "display_type": 1,
    }
    data = _get(url, params)
    results = []
    for d in data[:limit]:
        results.append(
            {
                "name": d.get("name"),
                "address": d.get("display"),
                "distance_km": round(d.get("distance", 0), 3),
            }
        )
    return results


# ===============================================================
# 3️⃣ ENRICH 1 ĐỊA CHỈ
# ===============================================================
def enrich_address_with_vietmap(address: str, label: str):
    """Lấy lat/lng và các tiện ích xung quanh 1 địa chỉ."""
    try:
        detail = search_address(address)
        if not detail:
            return {"label": label, "address": address, "error": "Không tìm thấy địa chỉ"}

        lat, lng = detail["lat"], detail["lng"]
        grouped = get_nearby_groups(lat, lng)

        return {
            "label": label,
            "address": address,
            "lat": lat,
            "lng": lng,
            "city": detail.get("city"),
            "district": detail.get("district"),
            "ward": detail.get("ward"),
            "nearby_groups": grouped,  # 🌟 nhóm tiện ích chi tiết
        }
    except Exception as e:
        return {"label": label, "address": address, "error": str(e)}


# ===============================================================
# 4️⃣ ENRICH DÒNG CHAT MỚI NHẤT
# ===============================================================
def enrich_last_chat_record():
    """
    Đọc dòng cuối cùng trong data/chat_history.jsonl,
    trích địa chỉ từ answer, gọi VietMap enrich,
    rồi ghi đè lại đúng dòng đó.
    """
    path = "data/chat_history.jsonl"
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return

    try:
        last = json.loads(lines[-1])
    except Exception:
        return

    # Nếu đã enrich rồi thì bỏ qua
    if "properties" in last and last["properties"]:
        return

    answer = last.get("answer", "")
    pattern = r"📍\s*([^🏠📏💰📜\n]+)"
    matches = re.findall(pattern, answer or "")
    props = []

    for i, m in enumerate(matches, 1):
        addr = m.strip()
        if "Hà Nội" not in addr:
            addr += ", Hà Nội"
        props.append(enrich_address_with_vietmap(addr, f"Căn số {i}"))

    last["properties"] = props

    # Ghi đè lại dòng cuối
    lines[-1] = json.dumps(last, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"✅ Đã enrich VietMap cho dòng timestamp: {last.get('timestamp')}")
