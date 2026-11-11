import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import Layout from "./components/Layout";
import { ToastProvider } from "./components/ui";
import Overview from "./pages/Overview";
import NewRun from "./pages/NewRun";
import Runs from "./pages/Runs";
import RunDetail from "./pages/RunDetail";
import Datasets from "./pages/Datasets";
import Playground from "./pages/Playground";
import Compute from "./pages/Compute";
import Cleanup from "./pages/Cleanup";
import SettingsPage from "./pages/Settings";

try {
  document.documentElement.dataset.theme = localStorage.getItem("snapcall-theme") || "dark";
} catch {
  document.documentElement.dataset.theme = "dark";
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <ToastProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/new" element={<NewRun />} />
            <Route path="/runs" element={<Runs />} />
            <Route path="/runs/:id" element={<RunDetail />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/compute" element={<Compute />} />
            <Route path="/cleanup" element={<Cleanup />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Overview />} />
          </Routes>
        </Layout>
      </ToastProvider>
    </BrowserRouter>
  </StrictMode>,
);
