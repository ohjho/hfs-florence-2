import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import subprocess
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
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

def convert_to_od_format(data):
    bboxes = data.get('bboxes', [])
    labels = data.get('bboxes_labels', [])
    od_results = {
        'bboxes': bboxes,
        'labels': labels
    }
    return od_results

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

def process_image(image, task_prompt, text_input=None):
    if task_prompt == '<CAPTION>':
        result = run_example(task_prompt, image)
        return result
    elif task_prompt == '<DETAILED_CAPTION>':
        result = run_example(task_prompt, image)
        return result
    elif task_prompt == '<MORE_DETAILED_CAPTION>':
        result = run_example(task_prompt, image)
        return result
    elif task_prompt == '<OD>':
        results = run_example(task_prompt, image)
        fig = plot_bbox(image, results['<OD>'])
        return fig
    elif task_prompt == '<DENSE_REGION_CAPTION>':
        results = run_example(task_prompt, image)
        fig = plot_bbox(image, results['<DENSE_REGION_CAPTION>'])
        return fig
    elif task_prompt == '<REGION_PROPOSAL>':
        results = run_example(task_prompt, image)
        fig = plot_bbox(image, results['<REGION_PROPOSAL>'])
        return fig
    elif task_prompt == '<CAPTION_TO_PHRASE_GROUNDING>':
        results = run_example(task_prompt, image, text_input)
        fig = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
        return fig
    elif task_prompt == '<REFERRING_EXPRESSION_SEGMENTATION>':
        results = run_example(task_prompt, image, text_input)
        output_image = copy.deepcopy(image)
        output_image = draw_polygons(output_image, results['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True)
        return output_image
    elif task_prompt == '<REGION_TO_SEGMENTATION>':
        results = run_example(task_prompt, image, text_input)
        output_image = copy.deepcopy(image)
        output_image = draw_polygons(output_image, results['<REGION_TO_SEGMENTATION>'], fill_mask=True)
        return output_image
    elif task_prompt == '<OPEN_VOCABULARY_DETECTION>':
        results = run_example(task_prompt, image, text_input)
        bbox_results = convert_to_od_format(results['<OPEN_VOCABULARY_DETECTION>'])
        fig = plot_bbox(image, bbox_results)
        return fig
    elif task_prompt == '<REGION_TO_CATEGORY>':
        results = run_example(task_prompt, image, text_input)
        return results
    elif task_prompt == '<REGION_TO_DESCRIPTION>':
        results = run_example(task_prompt, image, text_input)
        return results
    elif task_prompt == '<OCR>':
        result = run_example(task_prompt, image)
        return result
    elif task_prompt == '<OCR_WITH_REGION>':
        results = run_example(task_prompt, image)
        output_image = copy.deepcopy(image)
        output_image = draw_ocr_bboxes(output_image, results['<OCR_WITH_REGION>'])
        return output_image

css = """
  #output {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1><center>Florence-2 Demo<center><h1>")
    with gr.Tab(label="Florence-2 Image Captioning"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Picture")
                task_prompt = gr.Dropdown(choices=[
                    '<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>', '<OD>',
                    '<DENSE_REGION_CAPTION>', '<REGION_PROPOSAL>', '<CAPTION_TO_PHRASE_GROUNDING>',
                    '<REFERRING_EXPRESSION_SEGMENTATION>', '<REGION_TO_SEGMENTATION>',
                    '<OPEN_VOCABULARY_DETECTION>', '<REGION_TO_CATEGORY>', '<REGION_TO_DESCRIPTION>',
                    '<OCR>', '<OCR_WITH_REGION>'
                ], label="Task Prompt")
                text_input = gr.Textbox(label="Text Input (optional)")
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Output Text")
                output_img = gr.Image(label="Output Image")

        gr.Examples(
            examples=[
                ["image1.jpg", '<CAPTION>'],
                ["image1.jpg", '<OD>'],
                ["image1.jpg", '<OCR_WITH_REGION>']
            ],
            inputs=[input_img, task_prompt],
            outputs=[output_text, output_img],
            fn=process_image,
            cache_examples=True,
            label='Try examples'
        )

        submit_btn.click(process_image, [input_img, task_prompt, text_input], [output_text, output_img])

demo.launch(debug=True)