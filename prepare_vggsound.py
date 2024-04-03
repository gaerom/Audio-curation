import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import json

#import youtube_dl
import yt_dlp as youtube_dl
import librosa
from pydub import AudioSegment
#from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

import gc
import concurrent.futures
import threading

from dataclasses import dataclass

@dataclass
class VGGSoundEntity :
    idx: int = None
    vid: str = None
    url: str = None
    stttime: int = None
    endtime: int = None
    label: str = None
    train_test_split: str = None
    raw_audio_name: str = None
    result_file_name: str = None
    
    download_status: bool = False
    #download_error: str = ""
    youtube_problem: bool = False
    process_status: bool = False
    process_error: str = ""

    def __init__(self, row) :
        self.idx = row['idx']
        self.vid = row['YouTube ID']
        self.url = row['url']
        self.stttime = int(row["start seconds"])
        self.endtime = self.stttime + 10
        self.label = row['label']
        self.train_test_split = str(row['train/test split'])
        self.raw_audio_name = str(row['raw_audio_name'])
        self.result_file_name = str(row['result_file_name'])

    def toDict(self) :
        return {
            "idx": self.idx,
            "vid": self.vid,
            "url": self.url,
            "stt": self.stttime,
            #"endtime": self.endtime,
            "cls": self.label,
            "splt": self.train_test_split,
            #"name": self.raw_audio_name,
            #"result_file_name": self.result_file_name,

            "dstt": self.download_status,
            "yerr": self.youtube_problem,
            "pstt": self.process_status,
            #"process_error": self.process_error
        }
    

def download_and_process_video(
    metadata:VGGSoundEntity,
    retry = 3
) :
    url = metadata.url
    stttime = metadata.stttime
    endtime = metadata.endtime
    train_test_split = metadata.train_test_split
    raw_file_path = os.path.join(
        RAW_VIDEO_ROOT_PATH, str(train_test_split), str(metadata.raw_audio_name)
    )
    result_file_path = os.path.join(
        VGGSOUND_DATASET_PATH, str(train_test_split), str(metadata.result_file_name)
    )
    
    ydl_opts = {
        'format': 'best',
        #'postprocessors': [{
        #    #'key': 'FFmpegExtractAudio',
        #    'preferredcodec': 'webm',
        #    #'preferredquality': '320',
        #}],
        'outtmpl': raw_file_path, #[:-4], # remove .mp3 because youtube_dl will add it
        'quiet':True,
        'external_downloader_args': ['-loglevel', 'panic']
    }

    # download video at most $retry times if error is not due to youtube problem
    for i in range(retry) :
        with youtube_dl.YoutubeDL(ydl_opts) as ydl :
            try :
                ydl.download([url])
                metadata.download_status = True
                break
            except Exception as e :
                if "errno 2" in str(e).lower() : # problem is temporary
                    metadata.youtube_problem =  False
                else : # video cannot downloaded due to youtube problem
                    metadata.youtube_problem = True
                    part_file_name = raw_file_path + ".part"
                    os.remove(part_file_name)
                    break
    if not metadata.download_status :
        return
        
    for i in range(retry) :
        try :
            # Load your video
            video_clip = VideoFileClip(raw_file_path)
            trimmed_clip = video_clip.subclip(stttime, endtime)
            trimmed_clip.write_videofile(result_file_path, codec='libx264', audio_codec='aac')

            '''
            # Load the WebM video
            video_clip = VideoFileClip(raw_file_path, fps_source="fps")

            # Trim the video clip
            trimmed_clip = video_clip.subclip(stttime, endtime)

            # Save the trimmed video as an MP4 file
            trimmed_clip.write_videofile(result_file_path, codec='libx264')
            '''

            # Close the video clips
            video_clip.close()
            trimmed_clip.close()

            metadata.process_status = True

            os.remove(raw_file_path)

            break
        except Exception as e :

            print("error : ", e)

            metadata.process_status = False

def download_and_process_parallel(metadata_list, max_workers = 50) :
    log_data = []    
    # we will use ThreadPoolExecutor to download videos in parallel
    with concurrent.futures.ThreadPoolExecutor(
        max_workers = max_workers
    ) as executor :
        future_to_row = {
            executor.submit(
                download_and_process_video, metadata
            ) :  metadata for metadata in metadata_list
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_row),
            total = len(metadata_list)
        ) :
        #for future in concurrent.futures.as_completed(future_to_row) :
            #row = future_to_row[future]
            future_to_row.pop(future)
            gc.collect()


if __name__ == "__main__" :
    SLINK = "https://www.youtube.com/watch?v="

    # directory where raw mp3 file downloaded from youtube will be saved
    RAW_VIDEO_ROOT_PATH = "./vggsound_raw"

    # directory where processed mp3 file will be saved
    VGGSOUND_DATASET_PATH = "./vggsound"

    # audioset_meta_data.json file will save information about download status
    # whether download were succeed, and the error message if not
    VGGSOUND_METADATA_PATH = "./metadata/vggsound_meta_data.json"
    
    UNAVAILABLE_VIDEO_LOG_PATH = "./log/vggsound_unavailable_video_log.txt"

    MAX_WORKERS = 50

    if os.path.exists(VGGSOUND_METADATA_PATH) :
        with open(VGGSOUND_METADATA_PATH, "r") as fp :
            vggsound_metadata_list = json.load(fp)
    else :
        # vggsound.csv : https://www.robots.ox.ac.uk/~vgg/data/vggsound/
        vggsound_data = pd.read_csv(
            "vggsoundsync.csv",
            names=["YouTube ID", "start seconds", "label", "train/test split"],
            index_col=False,
        )

        vggsound_data["url"] = SLINK + vggsound_data["YouTube ID"]
        vggsound_data["idx"] = list(map(str, range(len(vggsound_data))))
        vggsound_data["raw_audio_name"] = vggsound_data["label"]+ '_' + vggsound_data["idx"] + ".mp4"
        vggsound_data["result_file_name"] = vggsound_data["label"]+ '_' + vggsound_data["idx"] + ".mp4"


        if os.path.exists(UNAVAILABLE_VIDEO_LOG_PATH) :
            with open(UNAVAILABLE_VIDEO_LOG_PATH, "r") as fp :
                unavailable_vid_list = list(set(
                    list(map(
                        lambda x : x.split("___")[0],
                        fp.read().split("\n")
                    ))
                ))
            vggsound_data = vggsound_data[~vggsound_data["YouTube ID"].isin(unavailable_vid_list)]



        vggsound_metadata_list = [
            VGGSoundEntity(row) for _, row in vggsound_data.iterrows()
        ]

        '''
        for metadata  in vggsound_metadata_list :
            metadata_dict = metadata.toDict()
            if os.path.exists(os.path.join(
                VGGSOUND_DATASET_PATH,
                metadata.train_test_split,
                metadata.result_file_name
            )) :
                metadata.download_status = True
                metadata.process_status = True
        '''

        # if some of audios already downloaded, change according metadata status
        for metadata in vggsound_metadata_list :
            raw_file_path = os.path.join(
                RAW_VIDEO_ROOT_PATH, metadata.train_test_split, metadata.raw_audio_name
            )
            result_file_path = os.path.join(
                VGGSOUND_DATASET_PATH, metadata.train_test_split, metadata.result_file_name
            )
            if os.path.exists(result_file_path) :
                metadata.download_status = True
                metadata.youtube_problem = False
                metadata.process_status = True
            elif os.path.exists(raw_file_path) :
                metadata.download_status = True



    # initialte error recorder
    stop_flag = threading.Event()

    
    while True :
        print("metata data list length : ", len(vggsound_metadata_list))

        metadata_list_to_process = list(filter(
            lambda x : (not x.download_status) and (not x.process_status) and (not x.youtube_problem),
            vggsound_metadata_list
        ))

        print("number of videos to process : ", len(metadata_list_to_process))


        # we will use ThreadPoolExecutor to download videos in parallel
        with concurrent.futures.ThreadPoolExecutor(MAX_WORKERS) as executor :
            future_to_row = {
                executor.submit(
                    download_and_process_video, metadata
                ) :  metadata for metadata in  metadata_list_to_process
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_row),
                total = len(metadata_list_to_process)
            ) :
                #data = future_to_row.pop(future)
                future_to_row.pop(future)
                gc.collect()



        print("metata data list length : ", len(vggsound_metadata_list))
        print(
            "number of videos to process : ",
            len(list(filter(
                lambda x : (not x.download_status) and (not x.process_status) and (not x.youtube_problem),
                vggsound_metadata_list
            )))
        )

        option = input("continue? (y/n) : ")
        if option == "n" :
            break

    with open(VGGSOUND_METADATA_PATH, "w") as fp :
        json.dump(
            [metadata.toDict() for metadata in vggsound_metadata_list],
            fp,
            indent=4,
        )