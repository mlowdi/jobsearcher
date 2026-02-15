# `jobsearcher` - sök jobb med Claude

Visste du att Arbetsförmedlingen har [flera fantastiska API:er](https://data.arbetsformedlingen.se/)? Inte jag heller, men när jag fick veta det fick jag en idé.

Jag hatar nämligen att scrolla Platsbanken. Så jag tänkte: kan inte Claude göra det här åt mig?

`jobsearcher` är ett enkelt litet automationsprojekt. En gång per dygn kör jag `run.sh`, som hämtar de senaste 24 timmarnas platsannonser med yrkesgrupp `2516` (IT-säkerhetsspecialister) och skriver eventuella platsannonser i ett läsbart format till `jobs.txt`. Sedan startas Claude Code, läser in platsannonserna och matchar dem mot ett Markdown-formaterat CV från `resume.md` och bedömer på en skala från 1-10 hur bra varje platsannons matchar, för att sedan skriva en tabell till `<datum>-results.md` med en sorterad lista av tjänster och URL:er till annonserna i Platsbanken.

Så kan jag starta varje dag med att kolla vilka tjänster inom min yrkesgrupp som har publicerats och hur väl de matchar min profil.

"Jobba smartare, inte hårdare" som man säger.