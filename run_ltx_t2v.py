import os

import gradio as gr

from ltx_engine import (
    GDRIVE_AVAILABLE,
    GDRIVE_FOLDER_ID,
    OUTPUT_DIR,
    Video_Generation,
    list_outputs,
    load_ltx_model,
    upload_video_to_gdrive,
)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1000px !important; margin: auto !important; }
.brand-header { text-align: center; background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); padding: 28px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 10px 25px rgba(102,126,234,0.3); }
.brand-title { color: white; font-size: 2em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { color: rgba(255,255,255,0.88); font-size: 1em; margin-bottom: 16px; }
.social-buttons { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }
.social-btn { padding: 10px 24px; border-radius: 8px; font-weight: 700; font-size: 15px; text-decoration: none; display: inline-block; color: white; transition: all 0.3s; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
.social-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.3); }
.youtube-btn { background: linear-gradient(135deg, #FF0000 0%, #CC0000 100%); }
.x-btn { background: linear-gradient(135deg, #000000 0%, #333333 100%); }
button.primary { background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#stop-btn { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#clear-btn { background: linear-gradient(135deg, #6b7280 0%, #374151 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
.footer { text-align: center; padding: 20px; margin-top: 30px; border-top: 2px solid #e5e7eb; color: #6b7280; }
"""


def delete_selected_output(selected):
    if not selected:
        return list_outputs(), "No video selected."
    path = os.path.join(OUTPUT_DIR, selected)
    if os.path.exists(path):
        try:
            os.remove(path)
            return list_outputs(), f"Deleted {selected}."
        except Exception as e:
            return list_outputs(), f"Delete failed: {e}"
    return list_outputs(), "File not found."


def delete_all_outputs():
    if not os.path.isdir(OUTPUT_DIR):
        return [], "No outputs to delete."
    for filename in os.listdir(OUTPUT_DIR):
        if filename.lower().endswith(('.mp4', '.mkv', '.webm')):
            try:
                os.remove(os.path.join(OUTPUT_DIR, filename))
            except Exception:
                pass
    return list_outputs(), "All output videos deleted."


def backup_video_to_gdrive(selected):
    if not selected:
        return "No video selected."
    path = os.path.join(OUTPUT_DIR, selected)
    if not os.path.exists(path):
        return "File not found."
    if not GDRIVE_AVAILABLE:
        return "Google Drive is not configured; local file retained."
    file_id = upload_video_to_gdrive(path, GDRIVE_FOLDER_ID)
    if not file_id:
        return "Drive backup failed; local file retained."
    return f"Backed up {selected} to Google Drive ({file_id})."


def backup_all_videos_to_gdrive():
    videos = list_outputs()
    if not videos:
        return "No output videos to back up."
    if not GDRIVE_AVAILABLE:
        return "Google Drive is not configured; local files retained."
    backed_up = 0
    for selected in videos:
        path = os.path.join(OUTPUT_DIR, selected)
        file_id = upload_video_to_gdrive(path, GDRIVE_FOLDER_ID)
        if file_id:
            backed_up += 1
    return f"Backed up {backed_up}/{len(videos)} output video(s) to Google Drive."


def delete_selected_with_gdrive_backup(selected):
    if not selected:
        return list_outputs(), "No video selected."
    path = os.path.join(OUTPUT_DIR, selected)
    if not os.path.exists(path):
        return list_outputs(), "File not found."
    if not GDRIVE_AVAILABLE:
        return list_outputs(), "Google Drive is not configured; local file retained."
    file_id = upload_video_to_gdrive(path, GDRIVE_FOLDER_ID)
    if not file_id:
        return list_outputs(), "Drive backup failed; local file retained."
    try:
        os.remove(path)
        return list_outputs(), f"Deleted {selected} after Drive backup ({file_id})."
    except Exception as e:
        return list_outputs(), f"Backed up but delete failed: {e}"


def delete_all_with_gdrive_backup():
    videos = list_outputs()
    if not videos:
        return [], "No output videos to delete."
    if not GDRIVE_AVAILABLE:
        return list_outputs(), "Google Drive is not configured; local files retained."
    deleted = 0
    for selected in videos:
        path = os.path.join(OUTPUT_DIR, selected)
        file_id = upload_video_to_gdrive(path, GDRIVE_FOLDER_ID)
        if file_id and os.path.exists(path):
            try:
                os.remove(path)
                deleted += 1
            except Exception:
                pass
    return list_outputs(), f"Deleted {deleted}/{len(videos)} output video(s) after Drive backup."


with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="LTX-2.3 22B Video Generator") as demo:
    gr.HTML(
        '<div class="brand-header">'
        '<div class="brand-title">LTX-2.3 22B Distilled 1.1 Q4 Video Generator</div>'
        '<div class="brand-subtitle">Single-video Gradio UI; use run_batch.py for unattended CSV batches.</div>'
        '<div class="social-buttons">'
        '<a href="https://youtube.com/@aiquestacademy" target="_blank" class="social-btn youtube-btn">YouTube</a>'
        '<a href="https://x.com/aiquestacademy" target="_blank" class="social-btn x-btn">X</a>'
        '</div></div>'
    )

    gr.Markdown(
        "**Two-stage distilled 1.1 pipeline**\n\n"
        "Leave image inputs empty for Text-to-Video. Add a start image for Image-to-Video. "
        "Add both start and end images for first+last frame interpolation.\n\n"
        "For CSV batch generation, stop this UI and run `python run_batch.py`."
    )

    with gr.Column():
        prompt = gr.Textbox(
            label="Prompt",
            lines=3,
            placeholder="A majestic eagle soaring over snowy mountain peaks at golden hour, cinematic, 4K...",
        )

        with gr.Accordion("Image to Video (Optional)", open=False):
            with gr.Row():
                input_image_start = gr.Image(type="filepath", label="Start Frame")
                input_image_end = gr.Image(type="filepath", label="End Frame")
            gr.Markdown(
                "Start Frame only: Image-to-Video.\n"
                "Both frames: first+last frame interpolation.\n"
                "Neither frame: Text-to-Video."
            )

        with gr.Row():
            seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
            duration_dropdown = gr.Dropdown(
                label="Duration",
                choices=[
                    "2 Seconds (49 frames)",
                    "3 Seconds (73 frames)",
                    "5 Seconds (121 frames)",
                    "8 Seconds (193 frames)",
                    "10 Seconds (241 frames)",
                    "15 Seconds (361 frames)",
                ],
                value="5 Seconds (121 frames)",
            )

        with gr.Row():
            resolution_dropdown = gr.Dropdown(
                label="Base Resolution",
                choices=["1080p", "720p", "540p", "480p"],
                value="720p",
            )
            aspect_ratio_dropdown = gr.Dropdown(
                label="Aspect Ratio",
                choices=["16:9 Landscape", "4:3 Standard", "1:1 Square", "3:4 Portrait", "9:16 Portrait"],
                value="16:9 Landscape",
            )

        guide_scale = gr.Slider(
            label="Prompt Strength (guide_scale)",
            minimum=1.0,
            maximum=8.0,
            step=0.5,
            value=3.0,
        )
        num_steps = gr.Slider(
            label="Diffusion Steps",
            minimum=2,
            maximum=8,
            step=1,
            value=8,
        )

        with gr.Row():
            gen_btn = gr.Button("Generate Video", variant="primary", size="lg", elem_id="gen-btn")
            stop_btn = gr.Button("Stop", variant="secondary", size="lg", elem_id="stop-btn")
            clear_btn = gr.Button("Clear", variant="secondary", size="lg", elem_id="clear-btn")

        video_out = gr.Video(label="Output")
        status_out = gr.Textbox(label="Status", interactive=False)

        with gr.Accordion("Output Manager", open=False):
            gr.Markdown(
                f"**Google Drive Status:** {'ENABLED' if GDRIVE_AVAILABLE else 'DISABLED'}\n\n"
                "Videos are retained locally unless Drive backup succeeds."
            )

            refresh_outputs_btn = gr.Button("Refresh Outputs")
            outputs_dropdown = gr.Dropdown(
                label="Generated Videos",
                choices=[],
                interactive=True,
            )

            with gr.Row():
                backup_btn = gr.Button("Backup to Google Drive", variant="primary")
                delete_output_btn = gr.Button("Delete with Backup", variant="stop")

            with gr.Row():
                backup_all_btn = gr.Button("Backup All to Drive")
                delete_all_btn = gr.Button("Delete All after Backup", variant="stop")

            delete_status = gr.Textbox(label="Status", interactive=False)

            refresh_outputs_btn.click(
                fn=list_outputs,
                outputs=[outputs_dropdown],
            )

            backup_btn.click(
                fn=backup_video_to_gdrive,
                inputs=[outputs_dropdown],
                outputs=[delete_status],
            )

            backup_all_btn.click(
                fn=backup_all_videos_to_gdrive,
                outputs=[delete_status],
            )

            delete_output_btn.click(
                fn=delete_selected_with_gdrive_backup,
                inputs=[outputs_dropdown],
                outputs=[outputs_dropdown, delete_status],
            )

            delete_all_btn.click(
                fn=delete_all_with_gdrive_backup,
                outputs=[outputs_dropdown, delete_status],
            )

        gen_event = gen_btn.click(
            fn=Video_Generation,
            inputs=[prompt, input_image_start, input_image_end, seed, duration_dropdown,
                    resolution_dropdown, aspect_ratio_dropdown, guide_scale, num_steps],
            outputs=[video_out, status_out],
        )
        stop_btn.click(fn=None, cancels=[gen_event])
        clear_btn.click(
            fn=lambda: (None, None, None, "", -1),
            outputs=[input_image_start, input_image_end, video_out, prompt, seed],
        )

    gr.HTML(
        '<div class="footer">'
        '<p style="font-size: 16px; margin: 5px 0;">LTX-2.3 22B Distilled 1.1 Q4_K_M</p>'
        '<p style="font-size: 13px; margin: 10px 0;">'
        '<a href="https://youtube.com/@aiquestacademy" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 10px;">YouTube</a> | '
        '<a href="https://x.com/aiquestacademy" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 10px;">X</a>'
        '</p></div>'
    )


def launch():
    print("Loading LTX model once for the Gradio session...")
    load_ltx_model()
    print("Launching Gradio...")
    demo.queue()
    demo.launch(share=True, inline=False, debug=True, show_error=True, max_threads=1, ssr_mode=False)


if __name__ == "__main__":
    launch()
