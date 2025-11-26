import gradio as gr
from PIL import Image
from utils.utils import predict_disease, get_model
from pathlib import Path

def gradio_predict(image):
    try:
        result = predict_disease(image)
        
        if 'error' in result:
            return f'Error: {result['error']}'
        
        formatted_results = []
        for disease, confidence in sorted(result.items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * confidence
            formatted_results.append(f'{disease}: {percentage:.2f}')
        return '\n'.join(formatted_results)
    except Exception as e:
        return f"Error processing image: {str(e)}"

example_image_path = Path(__file__).parent / "images" 


demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Disease Prediction Results", max_length=160),
    title="AgriScan - Plant Disease Detection",
    description="Upload an image of a Cassava leaf to detect diseases. Supported diseases: bacterial blight, brown spot, green mite, healthy, mosaic",
    examples=[
        f'{example_image_path}/cassava-mosaic.jpg',
        f'{example_image_path}/CBBD.jpg',
        f'{example_image_path}/healthy.jpeg'
    ]
)

if __name__ == "__main__":
    try:
        model, checkpoint = get_model()
        print("Testing model loading... Success!")
        
        dummy_image = Image.new('RGB', (240, 240), color='green')
        result = predict_disease(dummy_image)
        
        if "error" not in result:
            print("Testing prediction... Success!")
            print(f"Sample prediction: {max(result.items(), key=lambda x: x[1])}")
            
            print("Launching Gradio interface...")
            demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
        else:
            print(f"Prediction test failed: {result['error']}")
            
    except Exception as e:
        print(f"Error during testing: {e}")