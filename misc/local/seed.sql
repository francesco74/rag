-- Archivi di prova per il sistema in locale (solo alla prima creazione del volume MySQL).
INSERT INTO topics (topic_id, description, prompt, public_access) VALUES
  ('greenlees', 'Archivio Ian Greenlees', 'greenlees', TRUE),
  ('attiprovincia', 'Atti amministrativi', 'atti', FALSE);

INSERT INTO sub_topics (topic_id, sub_topic_id, description, description_long) VALUES
  ('greenlees', 'lettere', 'Lettere', 'Corrispondenza dell''archivio'),
  ('attiprovincia', 'determine', 'Determine', 'Determinazioni dirigenziali'),
  ('attiprovincia', 'decreti_presidenziali', 'Decreti presidenziali', 'Decreti del Presidente'),
  ('attiprovincia', 'decreti_deliberativi', 'Decreti deliberativi', 'Decreti deliberativi');

-- Il sito lo decide il dominio: i browser risolvono da soli *.localhost.
INSERT INTO chat_sites (hostname, topic_id, name, allow_date_filter) VALUES
  ('greenlees.localhost', 'greenlees', 'Archivio Ian Greenlees', FALSE),
  ('atti.localhost', 'attiprovincia', 'Atti amministrativi', TRUE);
