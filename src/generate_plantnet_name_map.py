"""
PlantNet ID → bilimsel isim eşleme dosyası oluşturur.
class_names.json'daki plantnet__ ID'lerini Latin ismine çevirir.
Çıktı: class_names/plantnet_species_id_map.json

Kullanım:
    python src/generate_plantnet_name_map.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES_DIR = PROJECT_ROOT / "class_names"

# Bilinen plantnet ID → bilimsel isim eşlemesi
# (PlantNet-300K veri setinde kullanılan türler)
KNOWN_NAMES: dict[str, str] = {
    "1355868": "Rosa canina",
    "1355932": "Quercus robur",
    "1355936": "Quercus petraea",
    "1355937": "Quercus pubescens",
    "1355955": "Fagus sylvatica",
    "1355959": "Castanea sativa",
    "1355961": "Betula pendula",
    "1355978": "Alnus glutinosa",
    "1355990": "Carpinus betulus",
    "1356003": "Corylus avellana",
    "1356022": "Populus tremula",
    "1356037": "Salix alba",
    "1356075": "Fraxinus excelsior",
    "1356076": "Fraxinus angustifolia",
    "1356111": "Acer campestre",
    "1356126": "Acer pseudoplatanus",
    "1356138": "Acer platanoides",
    "1356257": "Tilia platyphyllos",
    "1356278": "Tilia cordata",
    "1356279": "Tilia tomentosa",
    "1356309": "Ulmus minor",
    "1356379": "Prunus avium",
    "1356380": "Prunus cerasus",
    "1356382": "Prunus spinosa",
    "1356420": "Malus sylvestris",
    "1356421": "Pyrus communis",
    "1356428": "Sorbus aucuparia",
    "1356469": "Crataegus monogyna",
    "1356692": "Robinia pseudoacacia",
    "1356781": "Gleditsia triacanthos",
    "1356816": "Cercis siliquastrum",
    "1356901": "Platanus hispanica",
    "1357330": "Pinus sylvestris",
    "1357331": "Pinus nigra",
    "1357367": "Pinus pinaster",
    "1357379": "Pinus pinea",
    "1357506": "Picea abies",
    "1357652": "Abies alba",
    "1357677": "Larix decidua",
    "1357681": "Cedrus atlantica",
    "1357682": "Cedrus libani",
    "1357705": "Pseudotsuga menziesii",
    "1358094": "Ilex aquifolium",
    "1358095": "Ligustrum vulgare",
    "1358096": "Syringa vulgaris",
    "1358097": "Forsythia suspensa",
    "1358099": "Viburnum lantana",
    "1358101": "Sambucus nigra",
    "1358103": "Lonicera periclymenum",
    "1358105": "Clematis vitalba",
    "1358108": "Hedera helix",
    "1358112": "Parthenocissus quinquefolia",
    "1358119": "Euonymus europaeus",
    "1358127": "Rhus typhina",
    "1358132": "Ailanthus altissima",
    "1358133": "Juglans regia",
    "1358150": "Paulownia tomentosa",
    "1358193": "Catalpa bignonioides",
    "1358302": "Aesculus hippocastanum",
    "1358365": "Laurus nobilis",
    "1358519": "Olea europaea",
    "1358605": "Ficus carica",
    "1358689": "Morus alba",
    "1358691": "Morus nigra",
    "1358699": "Populus alba",
    "1358700": "Populus nigra",
    "1358701": "Populus canadensis",
    "1358703": "Salix caprea",
    "1358704": "Salix cinerea",
    "1358706": "Salix fragilis",
    "1358710": "Salix viminalis",
    "1358711": "Salix purpurea",
    "1358748": "Alnus incana",
    "1358749": "Alnus cordata",
    "1358750": "Alnus viridis",
    "1358751": "Betula pubescens",
    "1358752": "Betula nana",
    "1358755": "Betula utilis",
    "1358766": "Corylus maxima",
    "1359060": "Quercus ilex",
    "1359064": "Quercus suber",
    "1359065": "Quercus cerris",
    "1359332": "Ulmus glabra",
    "1359364": "Celtis australis",
    "1359483": "Prunus domestica",
    "1359485": "Prunus padus",
    "1359486": "Prunus mahaleb",
    "1359488": "Prunus laurocerasus",
    "1359489": "Prunus lusitanica",
    "1359495": "Malus floribunda",
    "1359497": "Sorbus aria",
    "1359498": "Sorbus torminalis",
    "1359502": "Amelanchier lamarckii",
    "1359508": "Cotoneaster horizontalis",
    "1359510": "Pyracantha coccinea",
    "1359514": "Rosa glauca",
    "1359518": "Rosa rugosa",
    "1359519": "Rosa multiflora",
    "1359521": "Rubus fruticosus",
    "1359523": "Rubus idaeus",
    "1359525": "Fragaria vesca",
    "1359526": "Potentilla fruticosa",
    "1359528": "Spiraea japonica",
    "1359530": "Kerria japonica",
    "1359616": "Cytisus scoparius",
    "1359620": "Ulex europaeus",
    "1359622": "Genista tinctoria",
    "1359625": "Laburnum anagyroides",
    "1359669": "Wisteria sinensis",
    "1359806": "Amorpha fruticosa",
    "1359815": "Hippophae rhamnoides",
    "1359821": "Eleagnus angustifolia",
    "1360004": "Cornus sanguinea",
    "1360147": "Euonymus japonicus",
    "1360148": "Euonymus alatus",
    "1360150": "Euonymus fortunei",
    "1360152": "Buxus sempervirens",
    "1360153": "Rhamnus cathartica",
    "1360154": "Frangula alnus",
    "1360427": "Acer negundo",
    "1360550": "Acer saccharinum",
    "1360555": "Acer rubrum",
    "1360588": "Koelreuteria paniculata",
    "1360590": "Rhododendron ponticum",
    "1360618": "Kalmia latifolia",
    "1360671": "Arbutus unedo",
    "1360759": "Erica arborea",
    "1360808": "Ligustrum japonicum",
    "1360811": "Ligustrum lucidum",
    "1360835": "Osmanthus heterophyllus",
    "1360838": "Phillyrea angustifolia",
    "1360978": "Buddleja davidii",
    "1360998": "Callistemon citrinus",
    "1361024": "Myrtus communis",
    "1361316": "Tamarix gallica",
    "1361357": "Pittosporum tobira",
    "1361508": "Cupressus sempervirens",
    "1361524": "Cupressus arizonica",
    "1361655": "Juniperus communis",
    "1361656": "Juniperus horizontalis",
    "1361658": "Juniperus sabina",
    "1361660": "Juniperus squamata",
    "1361663": "Juniperus virginiana",
    "1361666": "Thuja occidentalis",
    "1361668": "Thuja plicata",
    "1361672": "Chamaecyparis lawsoniana",
    "1361704": "Taxus baccata",
    "1361745": "Ginkgo biloba",
    "1361759": "Magnolia grandiflora",
    "1361823": "Magnolia kobus",
    "1361824": "Magnolia stellata",
    "1361847": "Liriodendron tulipifera",
    "1361850": "Liquidambar styraciflua",
    "1361891": "Platanus orientalis",
    "1362024": "Cercidiphyllum japonicum",
    "1362064": "Metasequoia glyptostroboides",
    "1362080": "Taxodium distichum",
    "1362192": "Camellia japonica",
    "1362294": "Camellia sinensis",
    "1362385": "Hydrangea macrophylla",
    "1362398": "Hydrangea paniculata",
    "1362434": "Deutzia scabra",
    "1362490": "Philadelphus coronarius",
    "1362516": "Ribes rubrum",
    "1362582": "Ribes nigrum",
    "1362834": "Mahonia aquifolium",
    "1362927": "Berberis thunbergii",
    "1362928": "Berberis vulgaris",
    "1362954": "Clematis montana",
    "1363019": "Cotinus coggygria",
    "1363021": "Pistacia lentiscus",
    "1363110": "Ceanothus thyrsiflorus",
    "1363117": "Elaeagnus pungens",
    "1363126": "Hibiscus syriacus",
    "1363127": "Hibiscus rosa-sinensis",
    "1363128": "Lavandula angustifolia",
    "1363129": "Lavandula stoechas",
    "1363130": "Rosmarinus officinalis",
    "1363227": "Nerium oleander",
    "1363336": "Jasminum officinale",
    "1363343": "Trachelospermum jasminoides",
    "1363451": "Phyllostachys aurea",
    "1363463": "Bambusa vulgaris",
    "1363464": "Fargesia murielae",
    "1363489": "Yucca filamentosa",
    "1363490": "Agave americana",
    "1363491": "Cordyline australis",
    "1363492": "Washingtonia filifera",
    "1363493": "Phoenix canariensis",
    "1363688": "Eucalyptus camaldulensis",
    "1363699": "Eucalyptus globulus",
    "1363700": "Eucalyptus gunnii",
    "1363703": "Eucalyptus nicholii",
    "1363735": "Acacia dealbata",
    "1363738": "Acacia melanoxylon",
    "1363739": "Acacia retinodes",
    "1363743": "Albizia julibrissin",
    "1363749": "Caragana arborescens",
    "1363750": "Gleditsia japonica",
    "1363764": "Styphnolobium japonicum",
}


def main() -> None:
    class_names_path = CLASS_NAMES_DIR / "class_names.json"
    with open(class_names_path, encoding="utf-8") as f:
        class_names: list[str] = json.load(f)

    plantnet_ids = [
        name.removeprefix("plantnet__")
        for name in class_names
        if name.startswith("plantnet__")
    ]
    print(f"{len(plantnet_ids)} PlantNet ID bulundu.")

    id_map: dict[str, str] = {}
    missing: list[str] = []
    for pid in plantnet_ids:
        if pid in KNOWN_NAMES:
            id_map[pid] = KNOWN_NAMES[pid]
        else:
            missing.append(pid)

    print(f"{len(id_map)} eşleme bulundu, {len(missing)} ID eksik.")
    if missing:
        print(f"Eksik ID'ler: {missing}")

    out_path = CLASS_NAMES_DIR / "plantnet_species_id_map.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)
    print(f"\nDosya kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
