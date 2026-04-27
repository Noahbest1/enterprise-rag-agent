import React from "react";
import ReactDOM from "react-dom/client";
import AdminDashboard from "./AdminDashboard";
import AgentChat from "./AgentChat";
import Landing from "./Landing";
import StreamingChat from "./StreamingChat";
import UploadShell from "./UploadShell";
import "./styles.css";

// Route switch via `?view=`:
//   (none)  -> Landing page (4-card homepage)
//   upload  -> UploadShell (drag-drop PDF/DOCX into a new/existing KB)
//   stream  -> RAG streaming chat (POST /answer/stream)
//   agent   -> Agent chat (POST /agent/chat, plan + specialist cards)
//   admin   -> Complaint admin dashboard (claim / reply, pushes to user via SSE)
const params = new URLSearchParams(window.location.search);
const view = params.get("view");

let Root: React.ComponentType;
if (view === "upload") Root = UploadShell;
else if (view === "agent") Root = AgentChat;
else if (view === "stream") Root = StreamingChat;
else if (view === "admin") Root = AdminDashboard;
else Root = Landing;

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
