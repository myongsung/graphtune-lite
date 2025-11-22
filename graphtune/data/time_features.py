import numpy as np

STEPS_PER_DAY = 288  # 5분 간격
WEEK_NUM = 7

def make_time_features(num_steps: int):
    all_idx = np.arange(num_steps)
    time_in_day = (all_idx % STEPS_PER_DAY) / float(STEPS_PER_DAY)
    day_idx = all_idx // STEPS_PER_DAY
    week_id = (day_idx % WEEK_NUM).astype(np.int64)
    return time_in_day.astype(np.float32), week_id
