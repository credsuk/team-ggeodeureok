"""Fail closed on incomplete or unmapped local administrative boundaries."""
import json
import math

NAMES = {'서울특별시','부산광역시','대구광역시','인천광역시','광주광역시','대전광역시',
         '울산광역시','세종특별자치시','경기도','강원특별자치도','충청북도','충청남도',
         '전북특별자치도','전라남도','경상북도','경상남도','제주특별자치도'}
ALIASES = {'강원도':'강원특별자치도','전라북도':'전북특별자치도'}

def validate(geo):
    features = geo.get('features', [])
    if geo.get('type') != 'FeatureCollection' or len(features) != 17:
        raise ValueError('17개 광역자치단체 경계 검증 실패')
    names = [ALIASES.get(f.get('properties', {}).get('name'), f.get('properties', {}).get('name')) for f in features]
    if set(names) != NAMES or len(set(names)) != 17:
        raise ValueError('광역자치단체 지역명 중복 또는 매핑 누락')
    for f in features:
        g = f.get('geometry', {})
        if g.get('type') not in ('Polygon', 'MultiPolygon'):
            raise ValueError('지원하지 않는 행정 경계 도형')
        polygons = [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates']
        if not polygons:
            raise ValueError('빈 경계 좌표')
        for polygon in polygons:
            if not polygon:
                raise ValueError('빈 다각형')
            for ring in polygon:
                if len(ring) < 4 or ring[0] != ring[-1]:
                    raise ValueError('닫히지 않은 행정 경계')
                for point in ring:
                    if (len(point) != 2 or not all(isinstance(n, (float, int)) and math.isfinite(n) for n in point)
                            or not (124 <= point[0] <= 133 and 32 <= point[1] <= 40)):
                        raise ValueError('WGS84 대한민국 경위도 범위 검증 실패')
    return geo

def load(path):
    if not path.is_file():
        return {'available':False,'reason':'로컬 행정 경계 자료 없음. 외부 다운로드가 네트워크 권한(WinError 10013)으로 차단되어 지역별 목록으로 표시합니다.'}
    try:
        return {'available':True,'geojson':validate(json.loads(path.read_text(encoding='utf-8')))}
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        return {'available':False,'reason':'지도 검증 실패: '+str(exc)}
