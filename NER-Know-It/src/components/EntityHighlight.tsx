import React from "react";
import type { NERResult } from "../types/ner";

const entityColors = {
  'PER': 'bg-blue-200 text-blue-800',
  'LOC': 'bg-green-200 text-green-800',
  'ORG': 'bg-purple-200 text-purple-800',
  'MISC': 'bg-yellow-200 text-yellow-800'
};

export function EntityHighlight({ result }: { result: NERResult }) {
  let text = result.text;
  let markedText: React.ReactNode[] = [];
  let lastIndex = 0;
  const sortedEntities = [...result.entities].sort((a, b) => a.start - b.start);

  for (const entity of sortedEntities) {
    if (entity.start > lastIndex) {
      markedText.push(
        <span key={`text-${lastIndex}`}>
          {text.substring(lastIndex, entity.start)}
        </span>
      );
    }
    const colorClass = entityColors[entity.entity as keyof typeof entityColors] || 'bg-gray-200 text-gray-800';
    const wikiUrl = `https://en.wikipedia.org/wiki/${encodeURIComponent(entity.text.replace(/ /g, '_'))}`;
    markedText.push(
      <a
        key={`entity-${entity.start}`}
        href={wikiUrl}
        target="_blank"
        rel="noopener noreferrer"
        className={`px-1 rounded-md mx-1 cursor-pointer underline ${colorClass}`}
        title={`See "${entity.text}" on Wikipedia`}
      >
        {entity.text}
      </a>
    );
    lastIndex = entity.end;
  }
  if (lastIndex < text.length) {
    markedText.push(
      <span key={`text-${lastIndex}`}>
        {text.substring(lastIndex)}
      </span>
    );
  }
  return <>{markedText}</>;
}