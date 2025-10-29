const chat = document.getElementById("chat");

function addMsg(text, cls) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function botTyping() {
  const div = document.createElement("div");
  div.className = `msg bot typing`;
  div.textContent = "ngetik...";
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function send() {
  const q = text.value.trim();
  if (!q) return;
  text.value = "";
  addMsg(q, "user");

  const typing = await botTyping();
  const res = await fetch(`/chat?q=${encodeURIComponent(q)}`);
  const data = await res.json();
  typing.remove();

  addMsg(data.response, "bot");
}

async function sendImage() {
  const file = imgInput.files[0];
  if (!file) return;

  addMsg("📷 Oke, gue liat bentar yaa…", "user");
  const typing = await botTyping();

  const form = new FormData();
  form.append("image", file);

  const res = await fetch(`/vision`, {
    method: "POST",
    body: form
  });

  const data = await res.json();
  typing.remove();
  addMsg(data.response, "bot");

  imgInput.value = "";
}
