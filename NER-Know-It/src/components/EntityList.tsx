import type { NERResult } from "../types/ner";

export function EntityList({ result }: { result: NERResult }) {
  return (
    <ul className="divide-y divide-gray-200">
      {result.entities.map((entity, index) => {
        const wikiUrl = `https://en.wikipedia.org/wiki/${encodeURIComponent(entity.text.replace(/ /g, '_'))}`;
        return (
          <li key={index} className="px-4 py-3">
            <div className="flex items-center justify-between">
              <div>
                <a
                  href={wikiUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium underline text-blue-700 hover:text-blue-900"
                  title={`See "${entity.text}" on Wikipedia`}
                >
                  {entity.text}
                </a>
                <span className="ml-2 text-sm text-gray-500">
                  (Positions {entity.start} - {entity.end})
                </span>
              </div>
              <span className={`px-2 py-1 rounded-md text-xs ${
                entity.entity === 'PER' ? 'bg-blue-200 text-blue-800' :
                entity.entity === 'LOC' ? 'bg-green-200 text-green-800' :
                entity.entity === 'ORG' ? 'bg-purple-200 text-purple-800' :
                'bg-yellow-200 text-yellow-800'
              }`}>
                {entity.entity}
              </span>
            </div>
          </li>
        );
      })}
      {result.entities.length === 0 && (
        <li className="px-4 py-3 text-gray-500">
          No entity found.
        </li>
      )}
    </ul>
  );
}