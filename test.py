from evaldet import Tracks
from evaldet.mot_metrics.clearmot import calculate_clearmot_metrics

gt = Tracks.from_mot('gt.txt')
hyp = Tracks.from_mot('test.txt')

res = calculate_clearmot_metrics(gt, hyp)
print(res)