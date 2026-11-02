# Plan du rapport et de la soutenance (version concise)

**Titre provisoire** : Détection de sujets émergents et évolution temporelle dans la presse locale lyonnaise (GDELT, 3 août – 1er novembre 2026).

| Partie | Contenu | Où trouver la matière |
|---|---|---|
| 1. Problématique et hypothèses | Les sujets des médias lyonnais évoluent-ils ? Peut-on repérer automatiquement les thèmes et les sujets émergents ? H1 : les événements collectifs produisent des pics détectables ; H2 : « fréquent » ≠ « émergent » ; H3 : les médias ont des profils thématiques distincts. | README, notebook 06 |
| 2. Données | GDELT (API DOC inutilisable → fichiers GKG « translation »), 13 998 articles, 3 médias exploitables sur 7, périmètre d'analyse 4 936 titres. | notebooks 02, 03 |
| 3. Méthode | Nettoyage et lemmatisation ; TF-IDF, K-Means (LSA), LDA ; score d'émergence hebdomadaire pondéré par média ; annotation manuelle et sensibilité. | notebooks 04, 05, 06 |
| 4. Résultats | Vocabulaire et expressions ; 14 thèmes ; 87 termes émergents, 14 événements (mouvement lycéen, émeutes du 17 octobre, affaire Deranque…) ; profils par média. | notebooks 04 à 07, `outputs/figures`, `outputs/tables` |
| 5. Limites | Une saison ; titres seuls ; 3 médias dont un régional dominant ; clusters instables ; annotation par un seul lecteur ; filtre lyonnais heuristique. | fin des notebooks 02, 05, 07 |
| 6. Perspectives | Embeddings ou BERTopic ; texte intégral autorisé ; période plus longue ; second annotateur. | notebook 07 |

**Soutenance (≈ 10 min)** : problème → données et surprise de l'API (1 diapo) → méthode (1) → 3 figures clés : `07_fig1`, `07_fig2`, `07_fig3` → limites (1) → questions.
