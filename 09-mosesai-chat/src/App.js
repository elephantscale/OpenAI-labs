import React, { useState } from "react";
import axios from 'axios';

const Chat = () => {
  
    const [messages, setMessages] = useState([]);
    const [newMessage, setNewMessage] = useState('');

    const handleNewMessageChange = (event) => {
        setNewMessage(event.target.value);
    };  
    
    const handleEnterKeyMessage = async (event) => {
        if(event.key === 'Enter')
        {
           await handleSendMessage();
        }
    };

    const [isLoading, setLoading] = useState(false);

    const loading = (loading) => {
        setLoading(loading);
    };

    const handleSendMessage = async () => {
        if(newMessage.trim() === '')
        {
            return;
        }
        loading(true);
        const userMessage = { user: 'Learner', text: newMessage };
        setMessages((prevMessages) => [...prevMessages, userMessage]);

        try {
            const response = await axios.post('http://localhost:8000/ask', {
                question: newMessage
            });
            const botMessage = { user: 'MosesAI', text: response.data.answer };
            setMessages((prevMessages) => [...prevMessages, botMessage]);          
        } catch (error) {
            console.log(error);
        }
        finally{
            loading(false);
        }

        // Clear the message input field after sending the message
        setNewMessage('');
    };

    return (
        <div>
            
            <div className="main">
                <div>
                    {messages.map((message, index) => (
                        <div key={index} className={ message.user == 'Learner' ? 'question' : 'answer'}>
                            <strong>{message.user}</strong>: {message.text}
                        </div>
                        
                    ))}
                    {isLoading &&
                        <div className="dots-container">
                            <span className="dot"></span>
                            <span className="dot"></span>
                            <span className="dot"></span>
                        </div>
                    }
                </div>
            </div>
            <div className="question-div">
                <input className="question-input" placeholder="Send a question" value={newMessage} onKeyDown={handleEnterKeyMessage} onChange={handleNewMessageChange} />
                <button className="question-button" onClick={handleSendMessage}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" class="bi bi-send" viewBox="0 0 16 16">
                        <path d="M15.854.146a.5.5 0 0 1 .11.54l-5.819 14.547a.75.75 0 0 1-1.329.124l-3.178-4.995L.643 7.184a.75.75 0 0 1 .124-1.33L15.314.037a.5.5 0 0 1 .54.11ZM6.636 10.07l2.761 4.338L14.13 2.576 6.636 10.07Zm6.787-8.201L1.591 6.602l4.339 2.76 7.494-7.493Z"/>
                    </svg>
                </button>
            </div>
            
        </div>
    );
};

export default Chat;
