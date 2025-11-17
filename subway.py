import streamlit as st
import pandas as pd
import heapq
from collections import defaultdict


# ----------------------------------------------------
# 1. 데이터 로딩 및 그래프 구축
# ----------------------------------------------------

@st.cache_data
def load_data():
    """CSV 파일을 로드하고 그래프 구조를 구축합니다."""
    try:
        # subway.csv 로드: 역과 역사이의 시간 정보
        df_subway = pd.read_csv('subway.csv')

        # subwayLocation.csv 로드: 역의 경위도 정보
        df_location = pd.read_csv('subwayLocation.csv')

    except FileNotFoundError:
        st.error("🚨 'subway.csv' 또는 'subwayLocation.csv' 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    # 1-1. 그래프(인접 리스트) 구축 (양방향 처리)
    # graph = {'A': [('B', 5), ('C', 3)], ...} 형태
    graph = defaultdict(list)

    # 역 간 이동 시간을 양방향으로 그래프에 추가합니다.
    for _, row in df_subway.iterrows():
        start = row['start_station']
        end = row['end_station']
        time = row['time_minutes']

        # 정방향
        graph[start].append((end, time))
        # 역방향 (일반적으로 지하철 이동 시간은 양방향 동일하다고 가정)
        graph[end].append((start, time))

    # 1-2. 위치 정보 딕셔너리 구축
    # location_dict = {'강남': (37.4979, 127.0276), ...} 형태
    location_dict = {
        row['station']: (row['latitude'], row['longitude'])
        for _, row in df_location.iterrows()
    }

    # 1-3. 전체 역 목록 (셀렉트 박스에 사용)
    all_stations = sorted(list(graph.keys()))

    return graph, location_dict, all_stations


# ----------------------------------------------------
# 2. 다익스트라(Dijkstra's) 알고리즘 구현
# ----------------------------------------------------

def dijkstra_shortest_path(graph, start, end):
    """
    다익스트라 알고리즘을 사용하여 출발역에서 도착역까지의 최단 시간을 계산합니다.

    Args:
        graph (dict): 인접 리스트 형태의 그래프
        start (str): 출발역 이름
        end (str): 도착역 이름

    Returns:
        tuple: (최단 시간(float), 최단 경로(list))
    """
    # 1. 초기화
    # 최단 거리를 저장할 딕셔너리. 초기값은 무한대(infinity)
    distances = {station: float('inf') for station in graph}
    distances[start] = 0

    # 경로를 추적할 딕셔너리
    previous_stations = {station: None for station in graph}

    # 우선순위 큐(Min-Heap) 초기화: (거리, 역) 순서로 저장
    pq = [(0, start)]

    while pq:
        # 현재까지 가장 짧은 거리를 가진 노드(역)을 꺼냄
        current_distance, current_station = heapq.heappop(pq)

        # 이미 처리된 노드이거나, 현재 꺼낸 거리가 이미 저장된 최단 거리보다 길면 무시
        if current_distance > distances[current_station]:
            continue

        # 현재 역과 연결된 모든 이웃 역을 순회
        for neighbor, weight in graph.get(current_station, []):
            distance = current_distance + weight

            # 새로운 경로가 더 짧으면 업데이트
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_stations[neighbor] = current_station
                heapq.heappush(pq, (distance, neighbor))

    # 2. 결과 경로 추적 및 반환
    path = []
    current = end

    # 도착역부터 출발역까지 역순으로 경로를 추적
    while current is not None:
        path.append(current)
        if current == start:
            break
        current = previous_stations.get(current)

    path.reverse()  # 경로를 출발역 -> 도착역 순으로 뒤집음

    # 출발역이 경로의 시작이 아니거나 도착역의 거리가 무한대이면 경로 없음
    if path[0] != start or distances[end] == float('inf'):
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

    # 출발역과 도착역 선택
    start_station = st.sidebar.selectbox("출발역을 선택하세요:", all_stations)
    end_station = st.sidebar.selectbox("도착역을 선택하세요:", all_stations, index=len(all_stations) - 1)

    # 검색 버튼
    if st.sidebar.button("경로 검색 시작"):
        if start_station == end_station:
            st.warning("출발역과 도착역이 같습니다. 다른 역을 선택해 주세요.")
            return

        # 다익스트라 알고리즘 실행
        total_time, shortest_path = dijkstra_shortest_path(graph, start_station, end_station)

        st.subheader("✅ 검색 결과")

        if total_time == float('inf'):
            st.error(f"'{start_station}'에서 '{end_station}'까지 연결된 경로를 찾을 수 없습니다.")
        else:
            # 결과 표시
            st.success(f"**총 소요 시간:** {total_time:.1f} 분")
            st.info(f"**최단 경로:** {' → '.join(shortest_path)}")

            # ---------------------
            # 지도 시각화
            # ---------------------
            st.subheader("🗺️ 경로 지도 시각화")

            # 경로에 포함된 역들의 좌표만 추출하여 DataFrame 생성
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

                # 경로 설명 (출발/도착역 강조)
                map_explanation = []
                for i, row in df_path.iterrows():
                    label = row['station']
                    if row['station'] == start_station:
                        label = f"**출발역 ({row['station']})**"
                    elif row['station'] == end_station:
                        label = f"**도착역 ({row['station']})**"

                    if i < len(df_path) - 1:
                        # 다음 역과의 이동 시간
                        next_station = df_path.iloc[i + 1]['station']
                        time_to_next = next(
                            (time for neighbor, time in graph[row['station']] if neighbor == next_station),
                            None
                        )
                        map_explanation.append(f"{label} (경유) → 다음역({next_station})까지 {time_to_next}분")
                    else:
                        map_explanation.append(label)

                st.markdown("#### 경로 요약")
                st.markdown("\n\n".join(map_explanation))

            else:
                st.warning("경로에 해당하는 위치 데이터를 찾을 수 없어 지도를 표시할 수 없습니다.")
    else:
        st.info("좌측 사이드바에서 출발역과 도착역을 선택하고 '경로 검색 시작' 버튼을 눌러주세요.")
        st.markdown("---")
        st.subheader("현재 데이터셋에 포함된 전체 역 위치")
        df_all_locations = pd.DataFrame(location_dict).T.reset_index()
        df_all_locations.columns = ['station', 'latitude', 'longitude']
        if not df_all_locations.empty:
            st.map(df_all_locations, latitude='latitude', longitude='longitude', zoom=11)


if __name__ == "__main__":
    app()