import ClientClearViewDemo from "@/components/ClientClearViewDemo";

export default function Home() {
  return (
    <main className="min-h-screen bg-neutral-50 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <header className="text-center space-y-2">
          <h1 className="text-4xl font-extrabold tracking-tight text-neutral-900">
            ClearView <span className="text-blue-600">AI</span> Console
          </h1>
          <p className="text-neutral-500">
            Multi-Aspect Sentiment Analysis &amp; Conflict Resolution with MSR
          </p>
        </header>

        <ClientClearViewDemo />
      </div>
    </main>
  );
}
