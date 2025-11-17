import streamlit as st
import pandas as pd
import heapq
from collections import defaultdict
import sys


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

    # 1. subway.csv 로드 시도 (header=None 옵션 추가)
    for enc in encodings:
        try:
            # header=None: CSV에 컬럼 이름이 없음을 명시
            df_subway_temp = pd.read_csv('subway.csv', encoding=enc, header=None)
            df_subway_temp.columns = SUBWAY_COLUMNS  # 수동으로 컬럼 이름 할당
            df_subway = df_subway_temp
            st.sidebar.success(f"subway.csv 파일이 {enc} 인코딩으로 성공적으로 로드되었습니다.")
            break
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            st.error("🚨 'subway.csv' 파일을 찾을 수 없습니다. 파일 경로를 확인해 주세요.")
            st.stop()
        except Exception as e:
            # 첫 번째 로드 시도에서 발생하는 모든 예외를 상세하게 표시
            st.error(f"subway.csv 로드 중 예상치 못한 오류 발생 ({enc}): {e}")
            st.stop()

    # 2. subwayLocation.csv 로드 시도 (header=None 옵션 추가)
    for enc in encodings:
        try:
            # header=None: CSV에 컬럼 이름이 없음을 명시
            df_location_temp = pd.read_csv('subwayLocation.csv', encoding=enc, header=None)
            df_location_temp.columns = LOCATION_COLUMNS  # 수동으로 컬럼 이름 할당
            df_location = df_location_temp
            st.sidebar.success(f"subwayLocation.csv 파일이 {enc} 인코딩으로 성공적으로 로드되었습니다.")
            break
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            st.error("🚨 'subwayLocation.csv' 파일을 찾을 수 없습니다. 파일 경로를 확인해 주세요.")
            st.stop()
        except Exception as e:
            st.error(f"subwayLocation.csv 로드 중 예상치 못한 오류 발생 ({enc}): {e}")
            st.stop()

    # 최종 검증: 두 파일 중 하나라도 로드에 실패했다면 중단
    if df_subway is None or df_location is None:
        st.error("🚨 데이터 파일 로드에 실패했습니다. 파일이 깨지지 않았거나, 인코딩 문제가 지속되는지 확인해주세요.")
        st.stop()

    # 데이터 정리: 혹시 모를 역 이름/위경도의 앞뒤 공백 제거 (매우 중요)
    df_subway['start_station'] = df_subway['start_station'].astype(str).str.strip()
    df_subway['end_station'] = df_subway['end_station'].astype(str).str.strip()
    df_location['station'] = df_location['station'].astype(str).str.strip()

    # -------------------------------------------------------------------------
    # 1-1. 그래프(인접 리스트) 구축 (양방향 처리)
    # -------------------------------------------------------------------------
    graph = defaultdict(list)

    for _, row in df_subway.iterrows():
        start = row['start_station']
        end = row['end_station']

        # 시간 컬럼이 숫자가 아닐 경우를 대비해 float으로 변환
        try:
            time = float(row['time_minutes'])
        except ValueError:
            # 비정상적인 데이터가 발견되면 오류 메시지 출력 후 중단
            st.error(f"🚨 'time_minutes' 컬럼에 숫자가 아닌 값('{row['time_minutes']}')이 포함되어 있습니다. 데이터를 정리해 주세요.")
            st.stop()

        graph[start].append((end, time))
        graph[end].append((start, time))  # 양방향 처리 (시간이 동일하다고 가정)

    # 1-2. 위치 정보 딕셔너리 구축
    # 중복되는 역 이름이 있을 수 있으나, 지도 표시를 위해 마지막 값을 사용합니다.
    location_dict = {
        row['station']: (float(row['latitude']), float(row['longitude']))
        for _, row in df_location.iterrows()
    }

    # 1-3. 전체 역 목록 (셀렉트 박스에 사용)
    all_stations = sorted(list(graph.keys()))

    return graph, location_dict, all_stations


# ----------------------------------------------------
# 2. 다익스트라(Dijkstra's) 알고리즘 구현
# ----------------------------------------------------

def dijkstra_shortest_path(graph, start, end):
    """다익스트라 알고리즘을 사용하여 출발역에서 도착역까지의 최단 시간을 계산합니다."""
    # 출발역이나 도착역이 그래프에 없으면 빈 경로 반환
    if start not in graph or end not in graph:
        return float('inf'), []

    distances = {station: float('inf') for station in graph}
    distances[start] = 0
    previous_stations = {station: None for station in graph}
    # 우선순위 큐 초기화 (거리, 역 이름)
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

    # 경로가 유효하지 않거나 연결되지 않은 경우
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
                st.warning("경로에 해당하는 위치 데이터를 찾을 수 없어 지도를 표시할 수 없습니다.")
    else:
        st.info("좌측 사이드바에서 출발역과 도착역을 선택하고 '경로 검색 시작' 버튼을 눌러주세요.")
        st.markdown("---")
        st.subheader("현재 데이터셋에 포함된 전체 역 위치")
        # 지도에 전체 역을 표시하기 위한 DataFrame 준비
        if location_dict:
            df_all_locations = pd.DataFrame(location_dict).T.reset_index()
            df_all_locations.columns = ['station', 'latitude', 'longitude']
            if not df_all_locations.empty:
                st.map(df_all_locations, latitude='latitude', longitude='longitude', zoom=11)
        else:
            st.info("위치 데이터(subwayLocation.csv)가 로드되지 않았습니다.")


if __name__ == "__main__":
    app()