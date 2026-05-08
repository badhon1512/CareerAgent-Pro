import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CareerAgent Pro",
  description: "Agentic job intelligence chat interface",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
