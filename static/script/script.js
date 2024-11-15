// txt based bot
function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    console.log(userInput);
    if (userInput.trim() === "") return;

    const chatWindow = document.getElementById("messages");

    // Append user message to chat
    const userMessage = document.createElement("div");
    userMessage.classList.add("user-message");
    userMessage.textContent = userInput;
    chatWindow.appendChild(userMessage);

    // Clear the input box
    document.getElementById("user-input").value = "";

    // Send the message to the Flask backend
    fetch("/api/chat", {
        method: "POST",
        body: JSON.stringify({ message: userInput }),
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(response => response.json())
    .then(data => {
        const botMessage = document.createElement("div");
        botMessage.classList.add("bot-message");
        botMessage.textContent = data.response;
        chatWindow.appendChild(botMessage);
        chatWindow.scrollTop = chatWindow.scrollHeight; 
    })
    .catch(error => {
        console.error("Error:", error);
    });
}


function sendImg() {
    const imageUpload = document.getElementById('image-upload').files[0];
    const messages = document.getElementById('messages');

    if (imageUpload) {
        // Append user message (image preview)
        const userMessage = document.createElement('div');
        userMessage.classList.add('message', 'user-message');

        const imgElement = document.createElement('img');
        imgElement.src = URL.createObjectURL(imageUpload);
        imgElement.classList.add('image-preview');
        userMessage.appendChild(imgElement);

        messages.appendChild(userMessage);

        // Send image to backend for analysis
        const formData = new FormData();
        formData.append('image', imageUpload);

        fetch('/analyze-image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const botMessage = document.createElement('div');
            botMessage.classList.add('message', 'bot-message');
            botMessage.textContent = data.response;

            messages.appendChild(botMessage);
            messages.scrollTop = messages.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
        });

        // Clear image input
        document.getElementById('image-upload').value = '';
    }
}
