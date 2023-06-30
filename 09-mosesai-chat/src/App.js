import React, { useState } from "react";
import axios from 'axios';

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [newMessage, setNewMessage] = useState('');

    const handleNewMessageChange = (event) => {
        setNewMessage(event.target.value);
    };

    const handleSendMessage = async (userQuestion) => {
        const userMessage = { user: 'Learner', text: newMessage };
        setMessages((prevMessages) => [...prevMessages, userMessage]);
        setNewMessage('');
        console.log("userQuestion " + userQuestion.question)
        try {
            const response = await axios.get('http://localhost:8000/shema/', {
                params: {
                    user_id: 'Learner',
                    message: userQuestion
                }
            });
            const botMessage = { user: 'MosesAI', text: response.data.answer };
            setMessages((prevMessages) => [...prevMessages, botMessage]);
        } catch (error) {
            console.log(error);
        }
    };

    return (
        <div>
            <div>
                {messages.map((message, index) => (
                    <div key={index}>
                        <strong>{message.user}</strong>: {message.text}
                    </div>
                ))}
            </div>
            <input value={newMessage} onChange={handleNewMessageChange} />
            <button onClick={handleSendMessage}>Send</button>
        </div>
    );
};

export default Chat;

