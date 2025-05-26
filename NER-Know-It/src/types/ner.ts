export type Entity = {
  start: number;
  end: number;
  entity: string;
  text: string;
};

export type NERResult = {
  text: string;
  entities: Entity[];
};