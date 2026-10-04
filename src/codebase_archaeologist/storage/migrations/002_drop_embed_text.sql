-- embed_text duplicated `text` plus a header that is fully derivable from the
-- other columns (see chunking.build_embed_text). Dropping it roughly halves the
-- bytes uploaded per chunk.
alter table chunks drop column embed_text;
