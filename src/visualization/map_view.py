from __future__ import annotations

from collections.abc import Sequence

from src.core.models import Evidence

# Representative latitude/longitude points for visualization, not geodetic analysis.
# Several councils use approximate administrative-center coordinates rather than the exact office pin.
COUNCIL_COORDS: dict[str, tuple[float, float]] = {
    "서울특별시의회": (37.5675494, 126.9765657),
    "부산광역시의회": (35.1804603, 129.0755538),
    "대구광역시의회": (35.8715291, 128.6012447),
    "인천광역시의회": (37.4563, 126.7052),
    "광주광역시의회": (35.1600165, 126.8504927),
    "대전광역시의회": (36.3501644, 127.3838358),
    "울산광역시의회": (35.5384, 129.3114),
    "세종특별자치시의회": (36.4800, 127.2890),
    "경기도의회": (37.2889164, 127.0542038),
    "강원특별자치도의회": (37.8860815, 127.7293289),
    "충청북도의회": (36.6357, 127.4913),
    "충청남도의회": (36.6588, 126.6728),
    "전북특별자치도의회": (35.8204242, 127.1098809),
    "전라남도의회": (34.8162517, 126.4641876),
    "경상북도의회": (36.5760, 128.5060),
    "경상남도의회": (35.2380, 128.6960),
    "제주특별자치도의회": (33.4996, 126.5312),
}

ALIASES = {
    "서울": "서울특별시의회", "서울시": "서울특별시의회", "서울특별시": "서울특별시의회", "서울시의회": "서울특별시의회",
    "부산": "부산광역시의회", "부산시": "부산광역시의회", "부산광역시": "부산광역시의회", "부산시의회": "부산광역시의회",
    "대구": "대구광역시의회", "대구시": "대구광역시의회", "대구광역시": "대구광역시의회", "대구시의회": "대구광역시의회",
    "인천": "인천광역시의회", "인천시": "인천광역시의회", "인천광역시": "인천광역시의회", "인천시의회": "인천광역시의회",
    "광주": "광주광역시의회", "광주시": "광주광역시의회", "광주광역시": "광주광역시의회", "광주시의회": "광주광역시의회",
    "대전": "대전광역시의회", "대전시": "대전광역시의회", "대전광역시": "대전광역시의회", "대전시의회": "대전광역시의회",
    "울산": "울산광역시의회", "울산시": "울산광역시의회", "울산광역시": "울산광역시의회", "울산시의회": "울산광역시의회",
    "세종": "세종특별자치시의회", "세종시": "세종특별자치시의회", "세종특별자치시": "세종특별자치시의회", "세종시의회": "세종특별자치시의회",
    "경기": "경기도의회", "경기도": "경기도의회",
    "강원": "강원특별자치도의회", "강원도": "강원특별자치도의회", "강원특별자치도": "강원특별자치도의회", "강원도의회": "강원특별자치도의회",
    "충북": "충청북도의회", "충청북도": "충청북도의회", "충북도의회": "충청북도의회",
    "충남": "충청남도의회", "충청남도": "충청남도의회", "충남도의회": "충청남도의회",
    "전북": "전북특별자치도의회", "전라북도": "전북특별자치도의회", "전북특별자치도": "전북특별자치도의회", "전라북도의회": "전북특별자치도의회",
    "전남": "전라남도의회", "전라남도": "전라남도의회", "전남도의회": "전라남도의회",
    "경북": "경상북도의회", "경상북도": "경상북도의회", "경북도의회": "경상북도의회",
    "경남": "경상남도의회", "경상남도": "경상남도의회", "경남도의회": "경상남도의회",
    "제주": "제주특별자치도의회", "제주도": "제주특별자치도의회", "제주특별자치도": "제주특별자치도의회", "제주도의회": "제주특별자치도의회",
}


def canonical_assembly(value: str) -> str | None:
    name = " ".join(value.strip().split())
    if name in COUNCIL_COORDS:
        return name
    return ALIASES.get(name)


def build_map_points(evidence: Sequence[Evidence]) -> list[dict[str, object]]:
    """Number councils by the chronological order of their first evidence in the timeline."""
    points: list[dict[str, object]] = []
    seen_assemblies: set[str] = set()
    for utterance_order, item in enumerate(evidence, start=1):
        assembly = canonical_assembly(item.assembly) if item.assembly else None
        coords = COUNCIL_COORDS.get(assembly) if assembly else None
        if not assembly or not coords or assembly in seen_assemblies:
            continue
        map_order = len(points) + 1
        role = "최초 관측" if map_order == 1 else "후속 논의"
        points.append({
            "latitude": coords[0],
            "longitude": coords[1],
            "assembly": assembly,
            "date": item.date,
            "role": role,
            "order": map_order,
            "utterance_order": utterance_order,
            "order_label": str(map_order),
            "color": [37, 99, 235, 255] if map_order == 1 else [249, 115, 22, 255],
            "source_id": item.id,
            "meeting_title": item.meeting_title,
        })
        seen_assemblies.add(assembly)
    return points


def build_numbered_map_deck(points: Sequence[dict[str, object]]):
    """Build a PyDeck map with numeric sequence labels over each council marker."""
    if not points:
        raise ValueError("지도에 표시할 의회 위치가 없습니다.")

    import pydeck as pdk
    from pydeck.types import String

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            id="council-circles",
            data=list(points),
            get_position="[longitude, latitude]",
            get_fill_color="color",
            get_line_color=[255, 255, 255, 255],
            get_radius=18,
            radius_units=String("pixels"),
            line_width_min_pixels=2,
            stroked=True,
            pickable=False,
        ),
        pdk.Layer(
            "TextLayer",
            id="discussion-order-numbers",
            data=list(points),
            get_position="[longitude, latitude]",
            get_text="order_label",
            get_color=[255, 255, 255, 255],
            get_size=27,
            size_units=String("pixels"),
            get_text_anchor=String("middle"),
            get_alignment_baseline=String("center"),
            pickable=False,
        ),
    ]
    latitude = sum(float(point["latitude"]) for point in points) / len(points)
    longitude = sum(float(point["longitude"]) for point in points) / len(points)
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=latitude, longitude=longitude, zoom=5.2),
        map_style=None,
    )
