import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
import spaces

import requests
import copy

from PIL import Image, ImageDraw, ImageFont 
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random
import numpy as np
import cv2

import subprocess
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda").eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


DESCRIPTION = "# [Florence-2 Video Demo](https://huggingface.co/microsoft/Florence-2-large)"

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

@spaces.GPU
def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def plot_bbox(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    ax.axis('off')
    return fig

def draw_polygons(image, prediction, fill_mask=False):
    draw = ImageDraw.Draw(image)
    scale = 1
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    return image

def draw_ocr_bboxes(image, prediction):
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                  "{}".format(label),
                  align="right",
                  fill=color)
    return image

def process_video(video_path, task_prompt, text_input=None):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if task_prompt == 'Caption':
            task_prompt = '<CAPTION>'
            result = run_example(task_prompt, image)
            output_frames.append(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        elif task_prompt == 'Detailed Caption':
            task_prompt = '<DETAILED_CAPTION>'
            result = run_example(task_prompt, image)
            output_frames.append(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        elif task_prompt == 'More Detailed Caption':
            task_prompt = '<MORE_DETAILED_CAPTION>'
            result = run_example(task_prompt, image)
                        output_frames.append(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        elif task_prompt == 'Object Detection':
            task_prompt = '<OD>'
            results = run_example(task_prompt, image)
            fig = plot_bbox(image, results['<OD>'])
            output_frames.append(cv2.cvtColor(np.array(fig_to_pil(fig)), cv2.COLOR_RGB2BGR))
        elif task_prompt == 'Referring Expression Segmentation':
            task_prompt = '<REF_SEG>'
            results = run_example(task_prompt, image, text_input)
            annotated_image = draw_polygons(image.copy(), results['<REF_SEG>'])
            output_frames.append(cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR))
        elif task_prompt == 'OCR':
            task_prompt = '<OCR>'
            results = run_example(task_prompt, image)
            annotated_image = draw_ocr_bboxes(image.copy(), results['<OCR>'])
            output_frames.append(cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR))
        else:
            raise ValueError(f"Unsupported task prompt: {task_prompt}")

    video.release()

    output_path = 'output_video.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in output_frames:
        out.write(frame)
    out.release()

    return output_path

task_prompts = ['Caption', 'Detailed Caption', 'More Detailed Caption', 'Object Detection', 'Referring Expression Segmentation', 'OCR']

with gr.Blocks(css="style.css") as demo:
    with gr.Group():
        with gr.Row():
            video_input = gr.Video(
                label='Input Video',
                format='mp4',
                source='upload',
                interactive=True
            )
        with gr.Row():
            select_task = gr.Dropdown(
                label='Task Prompt',
                choices=task_prompts,
                value=task_prompts[0],
                interactive=True
            )
            text_input = gr.Textbox(
                label='Text Input (optional)',
                visible=False
            )
            submit = gr.Button(
                label='Process Video',
                scale=1,
                variant='primary'
            )
    video_output = gr.Video(
        label='Florence-2 Video Demo',
        format='mp4',
        interactive=False
    )

    submit.click(
        fn=process_video,
        inputs=[video_input, select_task, text_input],
        outputs=video_output,
    )

demo.queue().launch()