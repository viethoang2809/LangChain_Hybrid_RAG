import os
import requests
from dotenv import load_dotenv

# === Load API key từ file .env ===
load_dotenv()
VIETMAP_API_KEY = os.getenv("VIETMAP_API_KEY")
BASE_URL = "https://maps.vietmap.vn/api"


#Lấy toạ độ
def get_coordinates(address: str):
    url = f"{BASE_URL}/search/v4"
    params = {"apikey": VIETMAP_API_KEY, "text": address, "display_type": 1}
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()
    if not data:
        raise ValueError(f"Không tìm thấy kết quả cho địa chỉ: {address}")
    ref_id = data[0].get("ref_id")
    return get_place_coordinates(ref_id)


def get_place_coordinates(ref_id: str):
    url = f"{BASE_URL}/place/v4"
    params = {"apikey": VIETMAP_API_KEY, "refid": ref_id}
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()
    return {"lat": data.get("lat"), "lng": data.get("lng")}


#Tìm quanh theo keyword
def search_nearby_by_keyword(lat: float, lng: float, keyword: str, max_results: int = 3):
    url = f"{BASE_URL}/search/v4"
    params = {
        "apikey": VIETMAP_API_KEY,
        "text": keyword,
        "focus": f"{lat},{lng}",
        "display_type": 1,
    }
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()

    # Lấy tối đa 3 kết quả gần nhất, lọc trùng
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


#Gom nhóm tiện ích
def get_nearby_places_grouped(address: str, max_results_per_group: int = 3):
    coords = get_coordinates(address)
    lat, lng = coords["lat"], coords["lng"]

    # Bộ từ khóa chính bạn muốn
    MAIN_KEYWORDS = {
        "hospital": "bệnh viện",
        "primary_school": "tiểu học",
        "secondary_school": "trung học",
        "university": "đại học",
        "supermarket": "siêu thị",
        "park": "công viên",
        "restaurant": "nhà hàng",
    }

    grouped_results = {}
    for group, keyword in MAIN_KEYWORDS.items():
        items = search_nearby_by_keyword(lat, lng, keyword, max_results=max_results_per_group)
        grouped_results[group] = items

    return {
        "input_address": address,
        "center": coords,
        "groups": grouped_results
    }


#Test quanh Đại học Bách Khoa
if __name__ == "__main__":
    address = "Đại học Bách Khoa Hà Nội"
    result = get_nearby_places_grouped(address)

    print(f"📍 Trung tâm: {result['center']}")
    for group, places in result["groups"].items():
        print(f"\n📌 {group.upper()} ({len(places)}):")
        for p in places:
            print(f"- {p['name']} ({p['distance_km']} km): {p['address']}")
