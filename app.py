import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Goat Detection & Heat Analysis",
    page_icon="🐐",
    layout="wide"
)

st.title("🐐 Goat Detection & Heat Analysis System")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["📸 Image Detection", "🎥 Video Analysis & Heat Detection"])

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = "./model.pt"
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize model
model = load_model()

if model is None:
    st.error("Failed to load the model. Please check the model path.")
    st.stop()

def analyze_video_for_heat(video_path, model, movement_threshold=5, tail_movement_threshold=3, heat_threshold=1.0, max_frames=None, save_output_video=False, original_filename="video"):
    """Analyze video for tail wagging and heat detection with comprehensive logging"""
    
    # Create logging containers
    log_container = st.container()
    
    def log_message(message, level="INFO"):
        with log_container:
            if level == "ERROR":
                st.error(f"🔴 {message}")
            elif level == "WARNING":
                st.warning(f"🟡 {message}")
            elif level == "SUCCESS":
                st.success(f"🟢 {message}")
            else:
                st.info(f"ℹ️ {message}")
    
    log_message("Starting video analysis...")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        log_message("Could not open video file", "ERROR")
        return None, "Error: Could not open video file"
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / fps if fps > 0 else 0
    
    log_message(f"Video properties - FPS: {fps:.2f}, Total frames: {total_frames}, Resolution: {width}x{height}, Duration: {video_duration:.2f}s")
    
    # Set processing limits
    if max_frames is None:
        if total_frames > 3000:  # Limit to ~2 minutes at 30fps
            max_frames = 3000
            log_message(f"Large video detected. Processing limited to first {max_frames} frames to prevent timeout", "WARNING")
        else:
            max_frames = total_frames
    
    frames_to_process = min(total_frames, max_frames)
    log_message(f"Will process {frames_to_process} frames")
    
    # Setup video writer if saving output - using temporary path initially
    temp_output_path = None
    final_output_path = None
    out = None
    if save_output_video:
        temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='_temp_processed.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        log_message(f"Temporary output video being created...")
    
    # Initialize variables
    tail_relative_positions = []
    frame_timestamps = []
    detection_confidences = []
    bbox_data = []
    prev_frame = None
    frame_count = 0
    detection_count = 0
    skipped_camera_movement = 0
    skipped_no_detection = 0
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        log_message("Starting frame-by-frame processing...")
        
        while frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                log_message(f"End of video reached at frame {frame_count}")
                break
            
            frame_count += 1
            current_time = frame_count / fps
            frame_timestamps.append(current_time)
            
            # Update progress every 50 frames or at key intervals
            if frame_count % 50 == 0 or frame_count in [1, 10, 100]:
                progress = frame_count / frames_to_process
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{frames_to_process} ({progress*100:.1f}%)")
                log_message(f"Processed {frame_count} frames - Detections: {detection_count}, Skipped (camera): {skipped_camera_movement}, Skipped (no tail): {skipped_no_detection}")
            
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Camera movement detection
            camera_movement_detected = False
            if prev_frame is not None:
                frame_diff = cv2.absdiff(gray, prev_frame)
                _, diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                movement_pixels = np.sum(diff_thresh) / 255
                movement_ratio = movement_pixels / gray.size * 100
                
                # Skip frame if camera movement is too large
                if movement_ratio > movement_threshold:
                    camera_movement_detected = True
                    tail_relative_positions.append(None)
                    detection_confidences.append(0)
                    bbox_data.append(None)
                    prev_frame = gray
                    skipped_camera_movement += 1
            
            # Detect tail with YOLO
            if not camera_movement_detected:
                try:
                    results = model(frame, verbose=False)
                    boxes = results[0].boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        detection_count += 1
                        # Get tail position and confidence
                        xyxy = boxes[0].xyxy[0].cpu().numpy()
                        conf = float(boxes[0].conf[0])
                        x1, y1, x2, y2 = xyxy
                        center_y = (y1 + y2) / 2
                        
                        # Store detection data
                        detection_confidences.append(conf)
                        bbox_data.append((x1, y1, x2, y2, conf))
                        
                        # Calculate relative movement compared to previous frame
                        if len(tail_relative_positions) > 0 and tail_relative_positions[-1] is not None:
                            relative_movement = center_y - tail_relative_positions[-1]
                            # Only record if movement exceeds threshold
                            if abs(relative_movement) > tail_movement_threshold:
                                tail_relative_positions.append(center_y)
                            else:
                                tail_relative_positions.append(tail_relative_positions[-1])
                        else:
                            tail_relative_positions.append(center_y)
                        
                        # Draw bounding box and info on frame
                        if save_output_video:
                            # Draw bounding box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            
                            # Add text with confidence and frame info
                            cv2.putText(frame, f'Tail: {conf:.2f}', (int(x1), int(y1)-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(frame, f'Time: {current_time:.2f}s', (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                    else:
                        tail_relative_positions.append(None)
                        detection_confidences.append(0)
                        bbox_data.append(None)
                        skipped_no_detection += 1
                        
                        # Add "No Detection" text if saving video
                        if save_output_video:
                            cv2.putText(frame, 'No Detection', (10, height-30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(frame, f'Time: {current_time:.2f}s', (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                except Exception as e:
                    log_message(f"Error processing frame {frame_count}: {str(e)}", "ERROR")
                    tail_relative_positions.append(None)
                    detection_confidences.append(0)
                    bbox_data.append(None)
                    skipped_no_detection += 1
            
            # Add camera movement indicator if saving video
            if save_output_video and camera_movement_detected:
                cv2.putText(frame, 'CAMERA MOVEMENT - SKIPPED', (10, height-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'Time: {current_time:.2f}s', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to output video
            if save_output_video and out is not None:
                out.write(frame)
            
            prev_frame = gray
            
            # Memory management - force garbage collection every 500 frames
            if frame_count % 500 == 0:
                import gc
                gc.collect()
    
    except Exception as e:
        if out is not None:
            out.release()
        cap.release()
        log_message(f"Critical error during processing: {str(e)}", "ERROR")
        return None, f"Error during processing: {str(e)}"
    
    finally:
        if out is not None:
            out.release()
        cap.release()
        progress_bar.empty()
        status_text.empty()
    
    log_message(f"Frame processing completed! Total: {frame_count}, Detections: {detection_count}, Camera skips: {skipped_camera_movement}, No detection: {skipped_no_detection}")
    
    # Clean data and calculate tail wags
    log_message("Analyzing tail movement data...")
    cleaned_y = [y for y in tail_relative_positions if y is not None]
    
    log_message(f"Data cleaning complete - Valid positions: {len(cleaned_y)} out of {len(tail_relative_positions)} total frames")
    
    if len(cleaned_y) > 10:  # Need minimum data points
        try:
            # Normalize data to remove drift
            cleaned_y = np.array(cleaned_y) - np.mean(cleaned_y)
            log_message(f"Data normalized - Mean removed, std: {np.std(cleaned_y):.2f}")
            
            # Detect peaks with stricter parameters
            peaks, properties = scipy.signal.find_peaks(cleaned_y, prominence=5, distance=5)
            num_wags = len(peaks)
            
            log_message(f"Peak detection complete - Found {num_wags} peaks (potential tail wags)")
            
            # Calculate frequency based on actual processed duration
            actual_duration = frames_to_process / fps if fps > 0 else 1
            frequency = num_wags / actual_duration
            
            # Determine heat status
            is_in_heat = frequency > heat_threshold
            
            log_message(f"Analysis complete - Frequency: {frequency:.3f} wags/sec, Heat status: {'IN HEAT' if is_in_heat else 'NOT IN HEAT'}", "SUCCESS")
            
            # Move video to appropriate directory if saving
            if save_output_video and temp_output_path and os.path.exists(temp_output_path):
                # Create output directories
                base_output_dir = "output"
                heat_dir = os.path.join(base_output_dir, "heat" if is_in_heat else "not-inheat")
                os.makedirs(heat_dir, exist_ok=True)
                
                # Create filename with timestamp and analysis results
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = original_filename.split('.')[0] if '.' in original_filename else original_filename
                result_filename = f"{filename_base}_{timestamp}_freq{frequency:.2f}_{'HEAT' if is_in_heat else 'NOHEAT'}.mp4"
                final_output_path = os.path.join(heat_dir, result_filename)
                
                try:
                    # Move file from temp location to final location
                    import shutil
                    shutil.move(temp_output_path, final_output_path)
                    log_message(f"Video saved to: {final_output_path}", "SUCCESS")
                except Exception as e:
                    log_message(f"Error saving video to directory: {str(e)}", "ERROR")
                    final_output_path = temp_output_path  # Fallback to temp path
            elif save_output_video:
                final_output_path = temp_output_path
            
            return {
                'video_duration': actual_duration,
                'total_frames': total_frames,
                'processed_frames': frames_to_process,
                'valid_detections': len(cleaned_y),
                'detection_count': detection_count,
                'skipped_camera': skipped_camera_movement,
                'skipped_no_detection': skipped_no_detection,
                'num_wags': num_wags,
                'frequency': frequency,
                'is_in_heat': is_in_heat,
                'tail_positions': cleaned_y,
                'peaks': peaks,
                'fps': fps,
                'detection_rate': (detection_count / frames_to_process * 100) if frames_to_process > 0 else 0,
                'frame_timestamps': frame_timestamps[:len(tail_relative_positions)],
                'detection_confidences': detection_confidences,
                'bbox_data': bbox_data,
                'output_video_path': final_output_path
            }, None
            
        except Exception as e:
            log_message(f"Error in signal processing: {str(e)}", "ERROR")
            return None, f"Error in signal processing: {str(e)}"
    else:
        log_message(f"Insufficient data for analysis - only {len(cleaned_y)} valid detections found", "WARNING")
        return {
            'video_duration': video_duration,
            'total_frames': total_frames,
            'processed_frames': frames_to_process,
            'valid_detections': len(cleaned_y),
            'detection_count': detection_count,
            'skipped_camera': skipped_camera_movement,
            'skipped_no_detection': skipped_no_detection,
            'num_wags': 0,
            'frequency': 0,
            'is_in_heat': False,
            'tail_positions': [],
            'peaks': [],
            'fps': fps,
            'detection_rate': (detection_count / frames_to_process * 100) if frames_to_process > 0 else 0,
            'frame_timestamps': frame_timestamps[:len(tail_relative_positions)],
            'detection_confidences': detection_confidences,
            'bbox_data': bbox_data,
            'output_video_path': final_output_path if save_output_video else None
        }, "Insufficient valid tail detections for analysis"

def process_image(image, model, conf_thresh, iou_thresh):
    """Process image with YOLO model and return results"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Run inference
    results = model(img_array, conf=conf_thresh, iou=iou_thresh)
    
    # Get annotated image
    annotated_img = results[0].plot()
    
    # Convert BGR to RGB for display
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    return annotated_img_rgb, results[0]

# TAB 1: IMAGE DETECTION
with tab1:
    st.markdown("Upload images to test your YOLO goat detection model")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose image files", 
        type=['png', 'jpg', 'jpeg'], 
        accept_multiple_files=True,
        key="image_uploader"
    )

    # Model parameters
    st.sidebar.header("Detection Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"Image {i+1}: {uploaded_file.name}")
            
            # Create columns for original and processed images
            col1, col2 = st.columns(2)
            
            # Load and display original image
            image = Image.open(uploaded_file)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_column_width=True)
            
            # Process image
            with st.spinner("Processing image..."):
                try:
                    processed_img, results = process_image(
                        image, model, confidence_threshold, iou_threshold
                    )
                    
                    with col2:
                        st.markdown("**Detection Results**")
                        st.image(processed_img, use_column_width=True)
                    
                    # Display detection statistics
                    detections = results.boxes
                    if detections is not None and len(detections) > 0:
                        st.success(f"✅ Found {len(detections)} detection(s)")
                        
                        # Create a table with detection details
                        detection_data = []
                        for j, box in enumerate(detections):
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            class_name = model.names[cls] if cls < len(model.names) else f"Class_{cls}"
                            
                            detection_data.append({
                                "Detection": j+1,
                                "Class": class_name,
                                "Confidence": f"{conf:.3f}",
                                "Confidence %": f"{conf*100:.1f}%"
                            })
                        
                        st.table(detection_data)
                    else:
                        st.warning("⚠️ No detections found")
                        
                except Exception as e:
                    st.error(f"Error processing image: {e}")
            
            st.divider()

    else:
        st.info("👆 Upload one or more images to start testing your model")
        
        # Show some example instructions
        st.markdown("""
        ### How to use:
        1. **Upload Images**: Use the file uploader above to select goat images
        2. **Adjust Parameters**: Use the sidebar to fine-tune detection thresholds
        3. **View Results**: Compare original images with detection results
        4. **Analyze Performance**: Check detection statistics and confidence scores
        
        ### Tips for better results:
        - Use clear, well-lit images of goats
        - Adjust confidence threshold based on your needs
        - Lower thresholds = more detections (but possibly more false positives)
        - Higher thresholds = fewer detections (but higher confidence)
        """)

# TAB 2: VIDEO ANALYSIS
with tab2:
    st.markdown("Upload videos to analyze tail wagging and detect heat status")
    
    # Video analysis parameters
    st.sidebar.header("Video Analysis Parameters")
    movement_threshold = st.sidebar.slider("Camera Movement Threshold", 1, 20, 5, 1)
    tail_movement_threshold = st.sidebar.slider("Tail Movement Threshold", 1, 10, 3, 1)
    heat_threshold = st.sidebar.slider("Heat Detection Threshold (wags/sec)", 0.1, 5.0, 1.0, 0.1)
    max_frames = st.sidebar.slider("Max Frames to Process (0 = auto limit)", 0, 5000, 0, 100)
    save_output_video = st.sidebar.checkbox("Save Processed Video with Detections", value=True)
    
    # Set max_frames to None if 0 is selected
    max_frames = None if max_frames == 0 else max_frames
    
    # Video file uploader
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="video_uploader"
    )
    
    if uploaded_video is not None:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        try:
            st.video(uploaded_video)
            
            if st.button("🔍 Analyze Video for Heat Detection"):
                with st.spinner("Analyzing video... This may take a while."):
                    # Get original filename for organized saving
                    original_filename = uploaded_video.name
                    
                    results, error = analyze_video_for_heat(
                        video_path, 
                        model, 
                        movement_threshold, 
                        tail_movement_threshold, 
                        heat_threshold,
                        max_frames,
                        save_output_video,
                        original_filename
                    )
                
                if error:
                    st.error(f"Analysis failed: {error}")
                elif results:
                    # Display results
                    st.header("🎯 Analysis Results")
                    
                    # Create columns for metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Video Duration", f"{results['video_duration']:.2f} sec")
                    
                    with col2:
                        st.metric("Valid Detections", f"{results['valid_detections']}")
                    
                    with col3:
                        st.metric("Tail Wags", f"{results['num_wags']}")
                    
                    with col4:
                        st.metric("Frequency", f"{results['frequency']:.2f} wags/sec")
                    
                    # Heat status
                    if results['is_in_heat']:
                        st.success("✅ Goat is detected as IN HEAT")
                    else:
                        st.info("❌ Goat is NOT in heat")
                    
                    # Detailed analysis
                    st.subheader("📊 Detailed Analysis")
                    
                    analysis_data = {
                        "Metric": [
                            "Video Duration (seconds)",
                            "Total Frames",
                            "Processed Frames", 
                            "Valid Tail Detections",
                            "Successful Detections",
                            "Skipped (Camera Movement)",
                            "Skipped (No Detection)",
                            "Tail Wags Detected",
                            "Wag Frequency (per second)",
                            "Heat Status",
                            "Detection Rate (%)"
                        ],
                        "Value": [
                            f"{results['video_duration']:.2f}",
                            results['total_frames'],
                            results['processed_frames'],
                            results['valid_detections'],
                            results.get('detection_count', 'N/A'),
                            results.get('skipped_camera', 'N/A'),
                            results.get('skipped_no_detection', 'N/A'),
                            results['num_wags'],
                            f"{results['frequency']:.3f}",
                            "IN HEAT" if results['is_in_heat'] else "NOT IN HEAT",
                            f"{results.get('detection_rate', 0):.1f}%"
                        ]
                    }
                    
                    st.table(pd.DataFrame(analysis_data))
                    
                    # Plot tail movement if data available
                    if len(results['tail_positions']) > 0:
                        st.subheader("📈 Tail Movement Analysis")
                        
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
                        
                        # Plot 1: Tail positions over time
                        time_axis = np.arange(len(results['tail_positions'])) / results['fps']
                        ax1.plot(time_axis, results['tail_positions'], 'b-', alpha=0.7, linewidth=2, label='Tail Position')
                        
                        # Mark detected peaks (wags)
                        if len(results['peaks']) > 0:
                            peak_times = results['peaks'] / results['fps']
                            peak_values = [results['tail_positions'][i] for i in results['peaks']]
                            ax1.plot(peak_times, peak_values, 'ro', markersize=10, label=f'Detected Wags ({len(results["peaks"])})')
                        
                        ax1.set_ylabel('Tail Position (normalized)')
                        ax1.set_title('Tail Movement Over Time')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Plot 2: Detection confidence over time
                        if 'detection_confidences' in results and len(results['detection_confidences']) > 0:
                            conf_time_axis = np.arange(len(results['detection_confidences'])) / results['fps']
                            ax2.plot(conf_time_axis, results['detection_confidences'], 'g-', alpha=0.7, linewidth=2, label='Detection Confidence')
                            ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% Confidence')
                            ax2.set_ylabel('Confidence')
                            ax2.set_xlabel('Time (seconds)')
                            ax2.set_title('Detection Confidence Over Time')
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)
                            ax2.set_ylim(0, 1)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Show processed video if available
                        if results.get('output_video_path') and os.path.exists(results['output_video_path']):
                            st.subheader("🎬 Processed Video with Detections")
                            
                            # Show save location info
                            video_path = results['output_video_path']
                            if video_path.startswith('output/'):
                                st.success(f"✅ Video saved to: `{video_path}`")
                                
                                # Show directory structure
                                directory = os.path.dirname(video_path)
                                filename = os.path.basename(video_path)
                                st.info(f"""
                                **Organized Save Location:**
                                - Directory: `{directory}/`
                                - Filename: `{filename}`
                                - Status: {'🔥 IN HEAT' if results['is_in_heat'] else '❄️ NOT IN HEAT'}
                                """)
                            
                            # Create download button for the processed video
                            with open(results['output_video_path'], 'rb') as video_file:
                                video_bytes = video_file.read()
                                st.download_button(
                                    label="📥 Download Processed Video",
                                    data=video_bytes,
                                    file_name=os.path.basename(results['output_video_path']),
                                    mime="video/mp4"
                                )
                            
                            # Display the processed video
                            st.video(results['output_video_path'])
                            
                            st.info("""
                            **Processed Video Legend:**
                            - 🟢 Green boxes: Successful tail detections with confidence scores
                            - 🔴 Red text: "No Detection" when tail is not found
                            - 🟠 Orange text: "Camera Movement - Skipped" when frame is ignored due to camera shake
                            - Frame counter and timestamp are shown in white text
                            """)
                            
                            # Show directory contents if available
                            try:
                                output_dir = os.path.dirname(results['output_video_path'])
                                if os.path.exists(output_dir):
                                    files_in_dir = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
                                    if files_in_dir:
                                        st.expander("📁 Files in this category").write(f"Found {len(files_in_dir)} videos in `{output_dir}/`")
                            except:
                                pass
                            
                        else:
                            st.warning("Processed video not available for display")
                    
                    # Classification explanation
                    st.subheader("🧠 Classification Logic")
                    st.markdown(f"""
                    **Heat Detection Criteria:**
                    - Frequency threshold: {heat_threshold} wags per second
                    - Detected frequency: {results['frequency']:.3f} wags per second
                    - Result: {'**IN HEAT**' if results['is_in_heat'] else '**NOT IN HEAT**'}
                    
                    **Analysis Parameters:**
                    - Camera movement threshold: {movement_threshold}
                    - Tail movement threshold: {tail_movement_threshold}
                    - Minimum peak prominence: 5
                    - Minimum peak distance: 5 frames
                    """)
        
        finally:
            # Clean up temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    else:
        st.info("👆 Upload a video file to start heat detection analysis")
        
        st.markdown("""
        ### Video Analysis Features:
        1. **Tail Detection**: Uses YOLO model to detect tail in each frame
        2. **Movement Compensation**: Filters out camera movement effects
        3. **Wag Counting**: Detects peaks in tail movement to count wags
        4. **Heat Classification**: Determines heat status based on wag frequency
        
        ### Supported Formats:
        - MP4, AVI, MOV, MKV
        
        ### Analysis Parameters:
        - **Camera Movement Threshold**: Higher values ignore more camera shake
        - **Tail Movement Threshold**: Minimum movement to count as tail motion
        - **Heat Threshold**: Minimum wags per second to classify as in heat
        """)

# Model info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Information")
if model:
    st.sidebar.info(f"Model loaded successfully")
    try:
        class_names = list(model.names.values())
        st.sidebar.write("**Classes:**")
        for name in class_names:
            st.sidebar.write(f"- {name}")
    except:
        st.sidebar.write("Classes: Unable to retrieve")

# Add footer
st.markdown("---")
st.markdown("*Built with Streamlit and YOLO for goat detection and heat analysis*")