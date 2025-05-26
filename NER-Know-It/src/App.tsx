import axios from 'axios';
import { useState } from 'react';
import { EntityHighlight } from './components/EntityHighlight';
import { EntityList } from './components/EntityList';
import type { NERResult } from './types/ner';

function App() {
  const [text, setText] = useState('');
  const [results, setResults] = useState<NERResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) {
      setError('Please enter text');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8080/predict', { text });
      setResults(response.data);
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Something was wrong during the prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center">
          <h1 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
            NER CoNNL-2003 English Dataset
          </h1>
          <p className="mt-3 text-xl text-gray-500">
            Identify the entities in your text!
          </p>
        </div>

        <div className="mt-10 bg-white shadow rounded-lg p-6">
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label 
                htmlFor="text" 
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Text here!
              </label>
              <textarea
                id="text"
                name="text"
                rows={4}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                placeholder="Write something here..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </div>
            
            <div className="flex justify-center">
              <button
                type="submit"
                className="inline-flex items-center px-4 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                disabled={loading}
              >
                {loading ? 'Thinking...' : 'Analyzing text'}
              </button>
            </div>
          </form>

          {error && (
            <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-md">
              {error}
            </div>
          )}

          {results && (
            <div className="mt-8">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Results</h2>
              
              <div className="bg-gray-50 p-4 rounded-md">
                <div className="mb-4 leading-relaxed">
                  <EntityHighlight result={results} />
                </div>
                
                <div className="mt-4 border-t border-gray-200 pt-4">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Details</h3>
                  <div className="flex flex-wrap gap-2">
                    <span className="px-2 py-1 bg-blue-200 text-blue-800 rounded-md text-xs">PER: Person</span>
                    <span className="px-2 py-1 bg-green-200 text-green-800 rounded-md text-xs">LOC: Place</span>
                    <span className="px-2 py-1 bg-purple-200 text-purple-800 rounded-md text-xs">ORG: Organization</span>
                    <span className="px-2 py-1 bg-yellow-200 text-yellow-800 rounded-md text-xs">MISC: Else</span>
                  </div>
                </div>
              </div>
              
              <div className="mt-6">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Entities found</h3>
                <div className="bg-white shadow overflow-hidden rounded-md">
                  <EntityList result={results} />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;