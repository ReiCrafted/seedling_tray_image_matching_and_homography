import json

with open('good_matches.json', 'r') as f:
    data = json.load(f)

count_1_9 = 0
count_10_30 = 0
count_31_plus = 0

for item in data:
    inliers = item['inliers']
    if 1 <= inliers <= 9:
        count_1_9 += 1
    elif 10 <= inliers <= 30:
        count_10_30 += 1
    elif inliers >= 31:
        count_31_plus += 1

print(f"Total Matches: {len(data)}")
print(f"1-9 Inliers:   {count_1_9}")
print(f"10-30 Inliers: {count_10_30}")
print(f"31+ Inliers:   {count_31_plus}")
