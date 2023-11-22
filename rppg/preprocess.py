import os, shutil
from glob import glob
import csv
import cv2
import numpy as np
from scipy.interpolate import interp1d


def extract_frames(root_dir: str, output_root_dir: str, verbose: bool=False) -> None:
    """
    Extract frames from all videos and save to new directory.
    :param root_dir: Root directory containing case subdirectories
    :param output_root_dir: Root directory to save frames
    :param verbose: Whether to print verbose output
    """
    
    # Get all case subdirectories
    case_dirs = [path for path in glob(os.path.join(root_dir, "*")) if os.path.isdir(path)]

    for case_dir in case_dirs:
        # Get case name from directory
        case = os.path.basename(case_dir)

        # Create output directory to store frames and labels
        output_case_dir = os.path.join(output_root_dir, case)
        os.makedirs(output_case_dir, exist_ok=True)

        # Open CSV to store labels
        csv_file = os.path.join(output_case_dir, "labels.csv")
        csv_header = ["frame", "hr"]
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

        # Load RR interval data
        rr_file = os.path.join(case_dir, f"{case}_Mobi_RR-intervals.rr")
        rr_data = np.loadtxt(rr_file)

        # Extract and resample RR interval data to 10 Hz
        interp = interp1d(rr_data[:, 0], rr_data[:, 1], kind="cubic")
        time = np.arange(rr_data[0, 0], rr_data[-1, 0], 0.1)
        rr_interval = interp(time)

        # Convert RR interval to HR (bpm)
        hr = 60 / rr_interval

        # Load video file
        video_file = os.path.join(case_dir, f"{case}_edited.avi")
        video_cap = cv2.VideoCapture(video_file)
        
        if verbose:
            print(f"Case: {case}")
            print(f"Original RR interval data length: {len(rr_data)}, resampled to {len(time)} points")
            print(f"Total frames: {video_cap.get(cv2.CAP_PROP_FRAME_COUNT)}\tDuration: {video_cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_cap.get(cv2.CAP_PROP_FPS)}")

        frame_count = 0
        discarded_frames = 0
        for i in range(len(time)):
            if verbose:
                if i == 0:
                    print(f"First frame: {frame_count}\tTime: {time[i]}\tHR: {hr[i]}")
                if i == len(time) - 1:
                    print(f"Last frame: {frame_count}\tTime: {time[i]}\tHR: {hr[i]}")

            # Read video frame at frame position based on time
            frame_pos = int(time[i] * video_cap.get(cv2.CAP_PROP_FPS))
            if frame_pos >= video_cap.get(cv2.CAP_PROP_FRAME_COUNT):
                discarded_frames += 1
                continue
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = video_cap.read()

            if ret:
                # Save frame as PNG
                frame_file = os.path.join(output_case_dir, f"frame_{frame_count:05d}.png")
                cv2.imwrite(frame_file, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                # Append label to CSV
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([frame_count, hr[i]])

                frame_count += 1
            else:
                if verbose:
                    print(f"ERROR: Could not read frame {frame_count}.\ttime: {time[i]}\thr: {hr[i]}\tframe_pos: {frame_pos}")
                continue
        
        if verbose:
            print(f"Discarded {discarded_frames} out of range frames")
            print(f"Extracted {frame_count} frames to {output_case_dir}")
            print()

        # Release video capture
        video_cap.release()


def extract_face_roi(root_dir: str, output_root_dir: str, roi_size: tuple, verbose: bool=False) -> None:
    """
    Extract face ROI from all frames using Haar cascade and save to new directory.
    :param root_dir: Root directory containing case subdirectories
    :param output_root_dir: Root directory to save face ROIs
    :param roi_size: Size of face ROI
    :param verbose: Whether to print verbose output
    """
    
    # Load pre-trained Haar cascade XML file for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Get all case subdirectories
    case_dirs = [path for path in glob(os.path.join(root_dir, "*")) if os.path.isdir(path)]

    for case_dir in case_dirs:
        # Get case name from directory
        case = os.path.basename(case_dir)

        # Create output directory to store faces and labels
        output_case_dir = os.path.join(output_root_dir, case)
        os.makedirs(output_case_dir, exist_ok=True)

        # Get all frame files in case directory
        frame_files = glob(os.path.join(case_dir, "*.png"))

        for frame_file in frame_files:
            # Load frame
            frame = cv2.imread(frame_file)

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using Haar cascade
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 1:
                # Retain only face with largest bounding box
                areas = [w * h for (_, _, w, h) in faces]
                x, y, w, h = faces[areas.index(max(areas))]

            elif len(faces) == 1:
                x, y, w, h = faces[0]
                
            else:
                # If no faces are detected, skip to next frame
                if verbose:
                    print(f'ERROR: No faces detected in {frame_file}')
                continue

            # Crop frame to face ROI
            face = frame[y:y+h, x:x+w]

            # Resize face ROI to size using bicubic interpolation
            face = cv2.resize(face, roi_size, interpolation=cv2.INTER_CUBIC)

            # Get original filename of frame
            frame_id = os.path.basename(frame_file)

            # Save face ROI to output directory
            face_file = os.path.join(output_case_dir, frame_id)
            cv2.imwrite(face_file, face)

        if root_dir != output_root_dir:
            # Copy labels file to output directory
            labels_file = os.path.join(case_dir, "labels.csv")
            shutil.copy(labels_file, output_case_dir)

        if verbose:
            print(f"Extracted faces from {case_dir} to {output_case_dir}")


def flip_augmentation(root_dir: str, output_root_dir: str, verbose: bool=False) -> None:
    """
    Flip all frames horizontally and save to new directory.
    :param root_dir: Root directory containing case subdirectories
    :param output_root_dir: Root directory to save flipped frames
    :param verbose: Whether to print verbose output
    """
    
    # Get all case subdirectories
    case_dirs = [path for path in glob(os.path.join(root_dir, "*")) if os.path.isdir(path)]

    for case_dir in case_dirs:
        # Get case name from directory
        case = os.path.basename(case_dir)

        # Create output directory to store frames and labels
        output_case_dir = os.path.join(output_root_dir, f"{case}_flipped")
        os.makedirs(output_case_dir, exist_ok=True)

        # Get all frame files in case directory
        frame_files = glob(os.path.join(case_dir, "*.png"))

        for frame_file in frame_files:
            # Load frame
            frame = cv2.imread(frame_file)

            # Flip frame horizontally
            frame = cv2.flip(frame, 1)

            # Get original filename of frame
            frame_name = os.path.basename(frame_file)

            # Save flipped frame to output directory
            frame_file = os.path.join(output_case_dir, frame_name)
            cv2.imwrite(frame_file, frame)

        if root_dir != output_root_dir:
            # Copy labels file to output directory
            labels_file = os.path.join(case_dir, "labels.csv")
            shutil.copy(labels_file, output_case_dir)

        if verbose:
            print(f"Flipped frames from {case_dir} to {output_case_dir}")


def frequency_morphing(root_dir: str, output_root_dir: str, frequency_multiplier: float, verbose: bool=False) -> None:
    """
    Resample HR data and extract frames at new frequency.
    :param root_dir: Root directory containing case subdirectories
    :param output_root_dir: Root directory to save frames and labels
    :param frequency_multiplier: Frequency multiplier
    :param verbose: Whether to print verbose output
    """

    # Get all case subdirectories
    case_dirs = [path for path in glob(os.path.join(root_dir, "*")) if os.path.isdir(path)]

    for case_dir in case_dirs:
        # Get case name from directory
        case = os.path.basename(case_dir)

        # Create output directory to store frames and labels
        output_case_dir = os.path.join(output_root_dir, f"{case}_morphed")
        os.makedirs(output_case_dir, exist_ok=True)

        # Load HR data
        csv_file = os.path.join(case_dir, "labels.csv")
        hr_data = np.loadtxt(csv_file, delimiter=',', skiprows=1)  # Assuming first row is header

        # Resample HR data based on frequency multiplier
        new_hr_data = np.zeros((int(len(hr_data) * frequency_multiplier), 2))
        new_hr_data[:, 0] = np.linspace(hr_data[0, 0], hr_data[-1, 0], len(new_hr_data))
        interp = interp1d(hr_data[:, 0], hr_data[:, 1], kind="cubic")
        new_hr_data[:, 1] = interp(new_hr_data[:, 0])

        # Load video file
        video_file = os.path.join(case_dir, f"{case}_edited.avi")
        video_cap = cv2.VideoCapture(video_file)
        fps = video_cap.get(cv2.CAP_PROP_FPS)

        frame_count = 0
        for i in range(len(new_hr_data)):
            if verbose:
                print(f"Processing frame: {frame_count}")

            # Calculate time corresponding to current frame
            time = new_hr_data[i, 0]

            # Calculate frame position based on time
            frame_pos = int(time * fps)
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = video_cap.read()

            if ret:
                # Save frame as PNG
                frame_file = os.path.join(output_case_dir, f"frame_{frame_count:05d}.png")
                cv2.imwrite(frame_file, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                # Append label to CSV
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([frame_count, new_hr_data[i, 1]])

                frame_count += 1

        if verbose:
            print(f"Processed {frame_count} frames for case {case}")
        
        # Release video capture
        video_cap.release()