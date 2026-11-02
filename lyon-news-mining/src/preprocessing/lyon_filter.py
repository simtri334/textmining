"""Filtre « zone lyonnaise » : règles explicites, volontairement simples et explicables.

Un article est retenu si au moins une règle s'applique (la première qui matche est enregistrée) :
  1. `prefix`  : (Le Progrès) le préfixe de lieu du titre est une commune/territoire du Rhône ;
  2. `title`   : le titre contient Lyon, Villeurbanne, Rhône (hors « Bouches-du-Rhône », « Auvergne-Rhône-Alpes »…),
                 Beaujolais, ou une grande commune de la métropole ;
  3. `gkg_loc` : GDELT a géolocalisé l'article à Lyon ou à Villeurbanne (champ V2Locations).
C'est une heuristique : elle est évaluée à la main dans le notebook 02 (précision et rappel sur échantillon).
"""
import re

# Préfixes de lieu (Le Progrès) relevant du Rhône / de la métropole de Lyon.
RHONE_PREFIXES = {p.lower() for p in [
    "Lyon", "Métropole de Lyon", "Rhône", "En direct Rhône", "Villeurbanne", "Vénissieux", "Bron", "Saint-Priest",
    "Villefranche-sur-Saône", "Rillieux-la-Pape", "Décines-Charpieu", "Oullins-Pierre-Bénite", "Caluire-et-Cuire",
    "Vaulx-en-Velin", "Dardilly", "Tassin-la-Demi-Lune", "Tarare", "Givors", "Sainte-Foy-lès-Lyon",
    "Charbonnières-les-Bains", "Saint-Fons", "Écully", "Belleville-en-Beaujolais", "Beaujolais", "Meyzieu",
    "Francheville", "Saint-Genis-Laval", "Brignais", "L'Arbresle", "Mions", "Corbas", "Feyzin", "Chassieu",
    "Genas", "Neuville-sur-Saône", "Craponne", "Limonest", "Saint-Cyr-au-Mont-d'Or", "Mornant", "Anse",
    "Monts d'Or", "Pierre-Bénite", "Oullins", "Irigny", "Saint-Genis-les-Ollières", "Lissieu", "Vaugneray",
]}

# Mots du titre. `Rhône` : on écarte les noms composés qui ne désignent pas le département.
_RHONE_COMPOSED = re.compile(
    r"(Bouches|Haut|Bas|Auvergne|Pays)[- ]?(du[- ])?Rh[ôo]ne|Rh[ôo]ne[- ]Alpes"   # autres départements / région
    r"|\w+-du-Rh[ôo]ne|canal du Rh[ôo]ne au Rhin|Rh[ôo]ne-Poulenc|vall[ée]e du Rh[ôo]ne", re.I)   # communes hors Rhône, canal du Jura/Doubs
# Sigles (OL, LOU, TCL, ASVEL) : recherchés en respectant la casse, sinon « ol », « lou » créent des faux positifs.
_CASE_SENSITIVE = re.compile(r"\bOL\b|\bLOU\b|\bTCL\b|\bASVEL\b|\bHCL\b")
_CASE_INSENSITIVE = re.compile(
    r"\bLyon(nais|naise|naises)?\b|\bVilleurbanne\b|\bRh[ôo]ne\b|\bBeaujolais\b|\bV[ée]nissieux\b|\bVaulx-en-Velin\b|"
    r"\bCaluire\b|\bSaint-Priest\b|\bD[ée]cines\b|\bRillieux\b|\bOullins\b|\bGivors\b|\bTassin\b|\bPart-Dieu\b|\bSYTRAL\b|"
    r"\bHospices Civils\b|\bVillefranche\b|\bCondrieu\b|\bQuincieux\b|\bMeyzieu\b|\bSainte-Foy-l[èe]s-Lyon\b|\b[ÉE]cully\b", re.I)
GKG_LOC_RE = re.compile(r"(?:^|;)\d#(Lyon|Villeurbanne)[,#]")


def lyon_flag(title: str, place_prefix, locations):
    """Renvoie la règle qui a retenu l'article ('prefix', 'title', 'gkg_loc') ou None."""
    if isinstance(place_prefix, str) and place_prefix.lower() in RHONE_PREFIXES:
        return "prefix"
    t = _RHONE_COMPOSED.sub(" ", title or "")
    if _CASE_INSENSITIVE.search(t) or _CASE_SENSITIVE.search(t):
        return "title"
    if isinstance(locations, str) and GKG_LOC_RE.search(locations):
        return "gkg_loc"
    return None
