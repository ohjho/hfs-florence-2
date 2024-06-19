import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image, ImageDraw
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random

# Load model and processor
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_example(task_prompt, image, text_input=None):
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
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
    plt.axis('off')
    plt.show()

def draw_polygons(image, prediction, fill_mask=False):
    draw = ImageDraw.Draw(image)
    colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red']
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = color if fill_mask else None
        for polygon in polygons:
            draw.polygon(polygon, outline=color, fill=fill_color)
            draw.text((polygon[0][0], polygon[0][1]), label, fill=color)
    image.show()

def gradio_interface(image, task_prompt, text_input):
    result = run_example(task_prompt, image, text_input)
    if task_prompt in ['<OD>', '<OPEN_VOCABULARY_DETECTION>']:
        plot_bbox(image, result)
    elif task_prompt in ['<REFERRING_EXPRESSION_SEGMENTATION>', '<REGION_TO_SEGMENTATION>']:
        draw_polygons(image, result, fill_mask=True)
    return result

with gr.Blocks() as demo:
    gr.Markdown("## Florence Model Advanced Tasks")
    with gr.Row():
        image_input = gr.Image(type="pil")
        task_input = gr.Dropdown(label="Select Task", choices=[
            '<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>', 
            '<OD>', '<DENSE_REGION_CAPTION>', '<REGION_PROPOSAL>', 
            '<CAPTION_TO_PHRASE_GROUNDING>', '<REFERRING_EXPRESSION_SEGMENTATION>',
            '<REGION_TO_SEGMENTATION>', '<OPEN_VOCABULARY_DETECTION>',
            '<REGION_TO_CATEGORY>', '<REGION_TO_DESCRIPTION>', '<OCR>', '<OCR_WITH_REGION>'
        ])
        text_input = gr.Textbox(label="Optional Text Input", placeholder="Enter text here if required by the task")
        submit_btn = gr.Button("Run Task")
        output = gr.Textbox(label="Output")
    
    submit_btn.click(fn=gradio_interface, inputs=[image_input, task_input, text_input], outputs=output)

demo.launch()