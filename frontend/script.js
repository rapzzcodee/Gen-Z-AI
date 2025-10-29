const chat = document.getElementById("chat");

function addMsg(text, cls) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function send() {
  const q = text.value;
  text.value = "";
  addMsg(q, "user");

  const res = await fetch(`http://localhost:8000/chat?q=${encodeURIComponent(q)}`);
  const data = await res.json();
  addMsg(data.response, "bot");
}

async function sendImage() {
  const file = imgInput.files[0];
  addMsg("📷 ngeliat gambar bentar yaa…", "user");

  const form = new FormData();
  form.append("image", file);

  const res = await fetch("http://localhost:8000/vision", {
    method: "POST",
    body: form
  });
  const data = await res.json();
  addMsg(data.response, "bot");
}
