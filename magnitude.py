"""
추가해야 할 내용
1. 현재는 각각의 audio에 대한 magnitude의 mean값만 계산해서 저장
    a. 각각의 audio에 대한 중간값을 추가로 저장해야 함.
2. 그리고 각각의 audio magnitude의 평균과 중간값을 구했다면, 전체 값에 대한 평균과 중간값도 구하기
"""

import os
import numpy as np 
import librosa
import librosa.display
import matplotlib.pyplot as plt

# audio_dir = '/mnt/storage1/test'
audio_dir = '/mnt/storage1/vgg_dataset/raw_audios'
output_file = 'max_amplitude_values.txt'
total_result = 'results.txt'

# audio file
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

max_magnitudes = {}

# 전체 파일에 대한 값들
all_mean = []
all_median = []
all_max = []


# 모든 audio에 대해 처리
for file in audio_files:
    y, sr = librosa.load(os.path.join(audio_dir, file)) # audio array, sampling rate(22050)
    fft_result = np.fft.fft(y)
    magnitude_spectrum = np.abs(fft_result) # magnitude(절댓값을 적용했으니까)
    # print(f'{file}: {magnitude_spectrum}') # 각 audio file마다 magnitude가 저장된 값 확인
    
    # 하나의 audio file -> mean, median 계산
    mean_magnitude = np.mean(magnitude_spectrum)
    median_magnitude = np.median(magnitude_spectrum)
    
    all_mean.append(mean_magnitude)
    all_median.append(median_magnitude)
    
    # print(f'{file}: Mean: {mean_magnitude}, Median: {median_magnitude}')
    
    max_magnitude = np.max(magnitude_spectrum)
    max_magnitudes[file] = max_magnitude
    all_max.append(max_magnitude) # 최댓값들 모아놓은 list
    
    print(f'{file}: {max_magnitude}') # 계산된 magnitude에서 최댓값 확인
    
# 전체에 대한 mean, median 계산
total_mean = np.mean(all_mean)
total_median = np.median(all_median)
total_max_mean = np.mean(all_max) 

# print(f'{file}: 전체 Mean: {total_mean}, 전체 Median: {total_median}')

        
with open(output_file, 'w') as f:
    for file, magnitude in max_magnitudes.items():
        f.write(f'{file}: {magnitude}\n')

with open(total_result, 'w') as f:
    f.write(f'전체 mean: {total_mean}, 전체 median: {total_median}, 최댓값의 mean: {total_max_mean}\n')

