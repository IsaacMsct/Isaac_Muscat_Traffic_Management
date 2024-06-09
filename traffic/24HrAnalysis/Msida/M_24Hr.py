import datetime
from collections import defaultdict

def parse_traffic_data(file_path):
    time_blocks = {
        (0, 5): "00:00-05:59",
        (6, 6): "06:00-06:59",
        (7, 7): "07:00-07:59",
        (8, 8): "08:00-08:59",
        (9, 9): "09:00-09:59",
        (10, 10): "10:00-10:59",
        (11, 11): "11:00-11:59",
        (12, 12): "12:00-12:59",
        (13, 13): "13:00-13:59",
        (14, 14): "14:00-14:59",
        (15, 15): "15:00-15:59",
        (16, 16): "16:00-16:59",
        (17, 17): "17:00-17:59",
        (18, 18): "18:00-18:59",
        (19, 19): "19:00-19:59",
        (20, 23): "20:00-23:59"
    }

    traffic_situation_order = ['no traffic', 'moderate traffic', 'heavy traffic']
    weekday_data = defaultdict(lambda: defaultdict(lambda: 'no traffic'))
    weekend_data = defaultdict(lambda: defaultdict(lambda: 'no traffic'))

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            timestamp_str = parts[0]
            timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            hour = timestamp.hour
            weekday = timestamp.weekday()
            
            for (start, end), block in time_blocks.items():
                if start <= hour <= end:
                    break
            
            for traffic_part in parts[1:]:
                direction, situation = traffic_part.split(':')
                situation = situation.strip().replace('light traffic', 'moderate traffic')
                direction = direction.strip().split()[-1]
                
                data = weekday_data if weekday < 5 else weekend_data
                
                # Check if the situation is recognized before comparing
                if situation not in traffic_situation_order:
                    continue

                current_situation = data[block][direction]
                if traffic_situation_order.index(situation) > traffic_situation_order.index(current_situation):
                    data[block][direction] = situation

    print("Weekdays:")
    print_summary_table(weekday_data)
    print("\nWeekends:")
    print_summary_table(weekend_data)

def print_summary_table(data):
    print(f"{'Time Period':<15}{'Direction Mater Dei':<20}{'Direction St Julians':<20}{'Direction Marsa':<20}{'Direction Hamrun':<20}")
    for block, situations in sorted(data.items()):
        mater_dei = situations['Mater Dei']
        st_julians = situations['St Julians']
        marsa = situations['Marsa']
        hamrun = situations['Hamrun']
        print(f"{block:<15}{mater_dei:<20}{st_julians:<20}{marsa:<20}{hamrun:<20}")

if __name__ == '__main__':
    log_file_path = 'D:\\school\\TrafficDetection\\venv\\msida_traffic_report.txt'
    parse_traffic_data(log_file_path)
