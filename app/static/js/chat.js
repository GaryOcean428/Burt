document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chatContainer');
    const chatHistory = document.getElementById('chatHistory');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const newChatBtn = document.getElementById('newChatBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const uploadButton = document.getElementById('uploadButton');

    function appendMessage(sender, message, modelInfo = null) {
        const messageElement = document.createElement('div');
        messageElement.className = `flex ${sender === 'user' ? 'justify-end' : 'justify-start'}`;

        const contentElement = document.createElement('div');
        contentElement.className = `max-w-2xl ${sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800'} rounded-lg p-4 shadow-md`;

        contentElement.innerHTML = marked.parse(message);

        if (modelInfo) {
            const modelInfoElement = document.createElement('div');
            modelInfoElement.className = 'text-xs text-gray-500 mt-2';
            modelInfoElement.textContent = modelInfo;
            contentElement.appendChild(modelInfoElement);
        }

        messageElement.appendChild(contentElement);
        chatContainer.appendChild(messageElement);
        chatContainer.scrollTop = chatContainer.scrollHeight;

        messageElement.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });

        const historyItem = document.createElement('div');
        historyItem.className = 'p-3 hover:bg-gray-100 rounded-lg cursor-pointer transition duration-150 ease-in-out';
        historyItem.innerHTML = `<p class="font-medium text-gray-800">${sender === 'user' ? 'You' : 'Agent99'}</p><p class="text-sm text-gray-500 truncate">${message}</p>`;
        chatHistory.insertBefore(historyItem, chatHistory.firstChild);
    }

    function sendQuery() {
        const message = userInput.value.trim();
        if (message) {
            appendMessage('user', message);
            userInput.value = '';
            loadingSpinner.classList.remove('hidden');

            fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: message }),
            })
            .then(response => response.json())
            .then(data => {
                loadingSpinner.classList.add('hidden');
                if (data.error) {
                    appendMessage('ai', `Error: ${data.error}`);
                    console.error('Error details:', data.details);
                } else {
                    const {metadata} = data;
                    const metadataString = `Model: ${metadata.model_used} | Task: ${metadata.task_type} | Complexity: ${metadata.task_complexity.toFixed(2)}`;
                    appendMessage('ai', data.response, metadataString);
                }
            })
            .catch(error => {
                loadingSpinner.classList.add('hidden');
                console.error('Error:', error);
                appendMessage('ai', 'An error occurred while processing your request. Please try again.');
            });
        }
    }

    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendQuery();
        }
    });

    sendButton.addEventListener('click', sendQuery);

    newChatBtn.addEventListener('click', function() {
        chatContainer.innerHTML = '';
        appendMessage('ai', 'Hello! How can I assist you today?');
    });

    ['How do I set up the development environment?', 'Can you explain the key features of Agent99?', 'What\'s the best way to integrate external APIs?', 'How can I optimize my code for performance?'].forEach(msg => {
        const historyItem = document.createElement('div');
        historyItem.className = 'p-3 hover:bg-gray-100 rounded-lg cursor-pointer transition duration-150 ease-in-out';
        historyItem.innerHTML = `<p class="font-medium text-gray-800">You</p><p class="text-sm text-gray-500 truncate">${msg}</p>`;
        chatHistory.appendChild(historyItem);
    });

    uploadButton.addEventListener('click', function() {
        // Placeholder for future file upload functionality
        console.log('File upload button clicked');
        // You can implement the file upload logic here in the future
    });
});
