import streamlit as st
import pandas as pd
import heapq
from collections import defaultdict


# ----------------------------------------------------
# 1. 데이터 로딩 및 그래프 구축
# ----------------------------------------------------

@st.cache_data
def load_data():
    """
    CSV 파일을 로드하고 그래프 구조를 구축합니다.
    KeyError 방지를 위해 'utf-8-sig' 인코딩을 우선 적용합니다.
    """
    try:
        # subway.csv 로드: 역과 역사이의 시간 정보
        # 'utf-8-sig'는 한글 CSV에서 흔한 BOM(Byte Order Mark) 문제를 해결해줍니다.
        df_subway = pd.read_csv('subway.csv', encoding='utf-8-sig')

        # subwayLocation.csv 로드: 역의 경위도 정보
        df_location = pd.read_csv('subwayLocation.csv', encoding='utf-8-sig')

    except FileNotFoundError:
        st.error("🚨 'subway.csv' 또는 'subwayLocation.csv' 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
        st.stop()
    except Exception as e:
        # 만약 utf-8-sig로도 실패한다면 다른 인코딩(예: 'cp949', 'euc-kr')을 시도하도록 사용자에게 안내
        st.error(f"""
        🚨 파일을 읽는 중 오류가 발생했습니다: {e}

        **💡 해결 가이드:**
        1. CSV 파일의 컬럼 이름(예: 'start_station', 'time_minutes')에 오타가 없는지 확인하세요.
        2. 만약 'utf-8-sig'로 해결되지 않았다면, 파일 저장 시 사용된 인코딩(예: 'cp949' 또는 'euc-kr')으로 `encoding` 파라미터를 변경해보세요.
        """)
        st.stop()

    # 1-1. 그래프(인접 리스트) 구축 (양방향 처리)
    # graph = {'A': [('B', 5), ('C', 3)], ...} 형태
    graph = defaultdict(list)

    # DataFrame 컬럼 이름이 확실하게 존재함을 가정하고 접근합니다.
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
    """
    # 1. 초기화
    distances = {station: float('inf') for station in graph}
    distances[start] = 0
    previous_stations = {station: None for station in graph}
    pq = [(0, start)]  # 우선순위 큐(Min-Heap) 초기화: (거리, 역) 순서로 저장

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

    # 2. 결과 경로 추적 및 반환
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
                            (time for neighbor, time in graph.get(row['station'], []) if neighbor == next_station),
                            None
                        )
                        map_explanation.append(f"**{label}** → 다음역({next_station})까지 {time_to_next}분")
                    else:
                        map_explanation.append(f"**{label}**")

                st.markdown("#### 경로 요약")
                st.markdown("<br>".join(map_explanation), unsafe_allow_html=True)

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