# Create a UI for interactive predictions using ipywidgets
def predict_sentiment(review):
    global device  # Ensure device is recognized within this function
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = best_model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = np.argmax(probabilities.cpu().numpy())
    return "Positive" if predicted_class == 1 else "Negative", probabilities[0][predicted_class].item()

text_input = widgets.Textarea(
    value='',
    placeholder='Type a review here...',
    description='Review:',
    disabled=False
)

button = widgets.Button(description="Predict")
output = widgets.Output()

def on_button_click(b):
    with output:
        output.clear_output()
        sentiment, probability = predict_sentiment(text_input.value)
        print(f"Sentiment: {sentiment} (Probability: {probability:.4f})")

button.on_click(on_button_click)
display(text_input, button, output)
