from deepface import DeepFace

# Define the local image path
image_path = "./pdp.jpg"  # Replace with your actual image path

try:
    # Perform facial analysis (age, gender, emotion, and race)
    analysis = DeepFace.analyze(image_path, actions=['age', 'gender'])

    # Extract results
    age = analysis[0]['age']
    gender = analysis[0]['dominant_gender']
   

    # Display the results
    print(f"Age: {age}")
    print(f"Gender: {gender}")


except Exception as e:
    print(f"Error occurred: {e}")
