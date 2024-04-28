# Content-based-Image-Search

# Team Members

- Rushi Shah (B21AI032)
- Sushant Ravva (B21CS084)
- Jaysukh Makvana (B21CS043)

# Usage

To use the image retrieval system, follow these steps:

1. Clone or download this repository to your local machine.
2. Navigate to the directory containing the `interface_clip.py` script.
3. Run the script using the following command:

```
streamlit run interface_clip.py
```

4. The Streamlit interface will launch in your default web browser.
5. Enter the path to the folder containing the images you want to search through in the provided text input field on the sidebar.
6. Input your search query or prompt in the text input field.
7. Click the "Search" button to perform the image search.
8. The top 5 images most relevant to your query will be displayed along with their paths.

# About Script

The `interface_clip.py` script performs content-based image search using CLIP. Here's a brief overview of its functionality:

- It loads the CLIP model and sets up necessary configurations.
- Provides a Streamlit interface for user interaction.
- Allows users to input the path to the image folder and their search prompts.
  Utilizes the CLIP model to retrieve the most relevant images based on the user's prompt.
- Displays the top 5 images along with their paths.

# Notes

- Ensure that the provided image folder contains images in a compatible format (e.g., JPEG, PNG).

- CLIP model loading might take some time depending on your hardware configuration.
