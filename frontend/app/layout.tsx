import "./globals.css";

export const metadata = {
  title: "AI Civilization",
  description: "Observation of a finite world",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="wrap">
          <h1>AI Civilization</h1>
          <p className="muted" style={{ marginTop: "-.25rem" }}>
            Read-only observation. The world decides what is true; this only shows it.
          </p>
          {children}
        </div>
      </body>
    </html>
  );
}
