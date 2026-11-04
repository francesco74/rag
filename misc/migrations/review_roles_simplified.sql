-- Ruoli ridotti a revisore (testo), archivista (testo e metadati) e admin.
-- Da eseguire UNA SOLA VOLTA: un secondo passaggio trasformerebbe i nuovi revisori
-- in archivisti. L'ordine conta: il vecchio "revisore" poteva già fare tutto.
--
--   revisore, catalogatore -> archivista   (il catalogatore guadagna il testo)
--   correttore, validatore -> revisore     (il validatore guadagna il testo)
--   lettore                -> nessun ruolo: non esiste più un ruolo di sola lettura,
--                             l'amministratore assegna un ruolo a chi deve ancora entrare
START TRANSACTION;

INSERT IGNORE INTO review_user_roles (user_id, role)
    SELECT user_id, 'archivista' FROM review_user_roles WHERE role IN ('revisore', 'catalogatore');
DELETE FROM review_user_roles WHERE role = 'revisore';

INSERT IGNORE INTO review_user_roles (user_id, role)
    SELECT user_id, 'revisore' FROM review_user_roles WHERE role IN ('correttore', 'validatore');
DELETE FROM review_user_roles WHERE role IN ('lettore', 'correttore', 'catalogatore', 'validatore');

COMMIT;
