const API_URL = "http://127.0.0.1:8000/query";

async function sendQuery() {
    const input = document.getElementById("userInput").value;

    if (!input.trim()) {
        alert("Please enter something.");
        return;
    }

    document.getElementById("response").innerHTML = "Processing...";

    try {
        const res = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: input })
        });

        const data = await res.json();

        document.getElementById("response").innerHTML = `
            <h3>Agent: ${data.agent_name}</h3>
            <p>${data.response}</p>
        `;
    } catch (err) {
        document.getElementById("response").innerHTML =
            "❌ Connection Error. Ensure backend is running on 127.0.0.1:8000";
    }
}
