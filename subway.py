import streamlit as st
import pandas as pd
import heapq
from collections import defaultdict
import re  # 정규 표현식 모듈 추가
import sys


# 역 이름에서 괄호 안의 내용(노선 번호)을 제거하는 함수
def clean_station_name(name):
    """지하철 역 이름에서 괄호와 그 안의 내용(예: '(1)')을 제거하고 공백을 정리합니다."""
    # 괄호와 그 안의 내용을 제거 (예: '서울역(1)' -> '서울역')
    cleaned_name = re.sub(r'\s*\([^)]*\)$', '', str(name)).strip()
    return cleaned_name


# ----------------------------------------------------
# 1. 데이터 로딩 및 그래프 구축 (CSV 파일에 헤더가 없는 경우 처리)
# ----------------------------------------------------

@st.cache_data
def load_data():
    """
    CSV 파일을 로드하고 그래프 구조를 구축합니다.
    두 파일 모두 헤더가 없음을 가정하고 'header=None' 및 수동 컬럼 할당을 적용합니다.
    """

    # 예상되는 컬럼 이름
    SUBWAY_COLUMNS = ['start_station', 'end_station', 'time_minutes']
    LOCATION_COLUMNS = ['station', 'latitude', 'longitude']

    # 인코딩 리스트: 가장 흔한 오류 원인을 순서대로 시도
    encodings = ['utf-8-sig', 'cp949', 'euc-kr']

    df_subway = None
    df_location = None

    # 1. subway.csv 로드 시도
    for enc in encodings:
        try:
            df_subway_temp = pd.read_csv('subway.csv', encoding=enc, header=None)
            df_subway_temp.columns = SUBWAY_COLUMNS
            df_subway = df_subway_temp
            st.sidebar.success(f"subway.csv 파일이 {enc} 인코딩으로 성공적으로 로드되었습니다.")
            break
        except Exception:
            continue

    # 2. subwayLocation.csv 로드 시도
    for enc in encodings:
        try:
            df_location_temp = pd.read_csv('subwayLocation.csv', encoding=enc, header=None)
            df_location_temp.columns = LOCATION_COLUMNS
            df_location = df_location_temp
            st.sidebar.success(f"subwayLocation.csv 파일이 {enc} 인코딩으로 성공적으로 로드되었습니다.")
            break
        except Exception:
            continue

    # 최종 검증 및 예외 처리
    if df_subway is None:
        st.error("🚨 'subway.csv' 파일을 찾거나 로드할 수 없습니다.")
        st.stop()
    if df_location is None:
        st.error("🚨 'subwayLocation.csv' 파일을 찾거나 로드할 수 없습니다.")
        st.stop()

    # **핵심 수정 부분:** 역 이름 표준화 (경로-위치 데이터 매칭을 위해 괄호 제거)
    df_subway['start_station'] = df_subway['start_station'].apply(clean_station_name)
    df_subway['end_station'] = df_subway['end_station'].apply(clean_station_name)
    df_location['station'] = df_location['station'].apply(clean_station_name)

    # -------------------------------------------------------------------------
    # 1-1. 그래프(인접 리스트) 구축 (양방향 처리)
    # -------------------------------------------------------------------------
    graph = defaultdict(list)

    for _, row in df_subway.iterrows():
        start = row['start_station']
        end = row['end_station']

        try:
            time = float(row['time_minutes'])
        except ValueError:
            st.error(f"🚨 'time_minutes' 컬럼에 숫자가 아닌 값('{row['time_minutes']}')이 포함되어 있습니다. 데이터를 정리해 주세요.")
            st.stop()

        graph[start].append((end, time))
        graph[end].append((start, time))  # 양방향 처리 (시간이 동일하다고 가정)

    # 1-2. 위치 정보 딕셔너리 구축 (노선 번호가 제거된 표준화된 이름 사용)
    location_dict = {}
    for _, row in df_location.iterrows():
        station_name = row['station']
        try:
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            location_dict[station_name] = (lat, lon)
        except ValueError:
            st.error(f"🚨 '{station_name}' 역의 위도/경도 값이 숫자가 아닙니다. 데이터를 정리해 주세요.")
            st.stop()

    # 1-3. 전체 역 목록 (셀렉트 박스에 사용)
    all_stations = sorted(list(graph.keys()))

    return graph, location_dict, all_stations


# ----------------------------------------------------
# 2. 다익스트라(Dijkstra's) 알고리즘 구현
# ----------------------------------------------------

def dijkstra_shortest_path(graph, start, end):
    """다익스트라 알고리즘을 사용하여 출발역에서 도착역까지의 최단 시간을 계산합니다."""
    if start not in graph or end not in graph:
        return float('inf'), []

    distances = {station: float('inf') for station in graph}
    distances[start] = 0
    previous_stations = {station: None for station in graph}
    pq = [(0, start)]

    while pq:
        current_distance, current_station = heapq.heappop(pq)

        if current_distance > distances[current_station]:
            continue

        for neighbor, weight in graph.get(current_station, []):
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_stations[neighbor] = current_station
                heapq.heappush(pq, (distance, neighbor))

    # 경로 역추적
    path = []
    current = end

    while current is not None:
        path.append(current)
        if current == start:
            break
        current = previous_stations.get(current)

    path.reverse()

    if not path or path[0] != start or distances[end] == float('inf'):
        return float('inf'), []

    return distances[end], path


# ----------------------------------------------------
# 3. Streamlit 앱 메인 로직
# ----------------------------------------------------

def app():
    st.set_page_config(page_title="지하철 최단 경로 검색 (다익스트라)", layout="wide")
    st.title("🚇 지하철 최단 경로 검색 앱")
    st.markdown("---")

    # 데이터 로드
    graph, location_dict, all_stations = load_data()

    # 사이드바 (입력)
    st.sidebar.header("경로 검색")

    if not all_stations:
        st.error("🚨 그래프에 역 정보가 없습니다. CSV 파일 내용을 확인해주세요.")
        return

    default_index_end = len(all_stations) - 1 if len(all_stations) > 1 else 0

    start_station = st.sidebar.selectbox("출발역을 선택하세요:", all_stations, index=0)
    end_station = st.sidebar.selectbox("도착역을 선택하세요:", all_stations, index=default_index_end)

    # 검색 버튼
    if st.sidebar.button("경로 검색 시작"):
        if start_station == end_station:
            st.warning("출발역과 도착역이 같습니다. 다른 역을 선택해 주세요.")
            return

        total_time, shortest_path = dijkstra_shortest_path(graph, start_station, end_station)

        st.subheader("✅ 검색 결과")

        if total_time == float('inf'):
            st.error(f"'{start_station}'에서 '{end_station}'까지 연결된 경로를 찾을 수 없습니다.")
        else:
            st.success(f"**총 소요 시간:** {total_time:.1f} 분")
            st.info(f"**최단 경로:** {' → '.join(shortest_path)}")

            st.subheader("🗺️ 경로 지도 시각화")

            # 경로에 포함된 역들의 좌표만 추출 (clean_station_name이 적용된 상태)
            path_coords = []
            for station in shortest_path:
                if station in location_dict:
                    lat, lon = location_dict[station]
                    path_coords.append({
                        'station': station,
                        'latitude': lat,
                        'longitude': lon
                    })

            df_path = pd.DataFrame(path_coords)

            if not df_path.empty:
                # Streamlit의 map 기능을 사용
                st.map(df_path, latitude='latitude', longitude='longitude', zoom=12)

                # 경로 요약
                map_explanation = []
                for i, row in df_path.iterrows():
                    label = row['station']

                    if i < len(df_path) - 1:
                        next_station = df_path.iloc[i + 1]['station']
                        # 그래프는 표준화된 이름으로 연결되어 있으므로, 이동 시간 조회가 가능
                        time_to_next = next(
                            (time for neighbor, time in graph.get(row['station'], []) if neighbor == next_station),
                            None
                        )
                        label_status = "출발역" if row['station'] == start_station else ""
                        map_explanation.append(
                            f"**{label_status}** {label} → 다음역({next_station})까지 **{time_to_next}분**")
                    else:
                        label_status = "도착역"
                        map_explanation.append(f"**{label_status}** {label}")

                st.markdown("#### 경로 요약")
                st.markdown("<br>".join(map_explanation), unsafe_allow_html=True)

            else:
                # 지도 시각화에 실패했을 때, 사용자에게 원인 정보 제공
                st.warning(f"""
                    경로에 해당하는 위치 데이터를 찾을 수 없어 지도를 표시할 수 없습니다.
                    <br>
                    **원인 추정:** 경로에 포함된 역 중 하나 이상이 
                    `subwayLocation.csv` 파일에 누락되어 있을 수 있습니다.
                    <br>
                    **찾지 못한 역:** {', '.join([s for s in shortest_path if s not in location_dict])}
                """, unsafe_allow_html=True)

    else:
        st.info("좌측 사이드바에서 출발역과 도착역을 선택하고 '경로 검색 시작' 버튼을 눌러주세요.")
        st.markdown("---")
        st.subheader("현재 데이터셋에 포함된 전체 역 위치")
        # 지도에 전체 역을 표시하기 위한 DataFrame 준비
        if location_dict:
            df_all_locations = pd.DataFrame(location_dict).T.reset_index()
            df_all_locations.columns = ['station', 'latitude', 'longitude']
            if not df_all_locations.empty:
                # 위도/경도가 숫자인지 확인 후 지도에 표시
                df_all_locations['latitude'] = pd.to_numeric(df_all_locations['latitude'], errors='coerce')
                df_all_locations['longitude'] = pd.to_numeric(df_all_locations['longitude'], errors='coerce')
                df_all_locations = df_all_locations.dropna(subset=['latitude', 'longitude'])

                if not df_all_locations.empty:
                    st.map(df_all_locations, latitude='latitude', longitude='longitude', zoom=11)
                else:
                    st.warning("위치 데이터의 위도/경도 값이 유효하지 않아 전체 지도를 표시할 수 없습니다.")
        else:
            st.info("위치 데이터(subwayLocation.csv)가 로드되지 않았습니다.")