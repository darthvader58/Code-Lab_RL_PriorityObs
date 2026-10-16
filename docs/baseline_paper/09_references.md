# 9. References

Status key: **✓✓** bibliographic details (venue, volume, pages/DOI) confirmed online during drafting
(Oct 2026) · **✓** source located, some details still to fill · **?** not yet located. Open each
source before citing it and convert to BibTeX in Overleaf.

## Dynamic targeting and onboard autonomy

- ✓✓ **[breitfeld2025learning]** Breitfeld, A., Candela, A., Delfa, J., Kangaslahti, A., Zilberstein,
  I., Chien, S., Wettergreen, D. *Learning-Based Planning for Improving Science Return of Earth
  Observation Satellites.* arXiv:2509.07997, 2025 (also International Workshop on Planning &
  Scheduling for Space, 2025). https://arxiv.org/abs/2509.07997
  — Key facts used: RL and imitation learning on a DP expert; IL +10.0%, RL +13.7% over best
  heuristic; scenarios: cloud avoidance (MODIS), storm hunting (GPM).
- ✓ **[dt2025cognisat6]** *Flight of Dynamic Targeting on the CogniSAT-6 Spacecraft.* arXiv:2509.05304,
  2025. https://arxiv.org/abs/2509.05304 — author list: copy from the paper.
  — Key facts used: JPL + Ubotica + Open Cosmos; onboard lookahead analysis and retargeting in a
  single overflight; flight demonstration July–September 2025 (per JPL slides). The "within 90
  seconds" figure comes from secondary press coverage; confirm it in the paper before quoting it.
- ✓ **[dt2025astra]** JPL Dynamic Targeting, ESA ASTRA 2025 extended abstract and slides.
  https://ai.jpl.nasa.gov/public/documents/papers/DT-Astra-2025-Extended-Abstract.pdf
- ✓ **[dt2025spaceops]** JPL Dynamic Targeting, SpaceOps 2025.
  https://ai.jpl.nasa.gov/public/documents/papers/dt-spaceops-2025.pdf
- ✓ **[breitfeld2024isairas]** Breitfeld, A., et al. i-SAIRAS 2024.
  https://ai.jpl.nasa.gov/public/documents/papers/breitfeld-isairas-2024.pdf
- ✓ **[chien2023dt]** Chien, S., et al. *Expanding and Maturing Dynamic Targeting.* ESA ASTRA 2023.
  https://robotics.estec.esa.int/ASTRA/Astra2023/2023%20PAPERS/27_Chien_Expanding%20and%20Maturing%20Dynamic%20Targeting.pdf
- ✓ **[ubotica_dt]** Ubotica press release on the JPL partnership (lookahead lead time 60–90 s in
  early experiments).
  https://www.ubotica.com/news/partnership-between-nasa-jpl-and-irish-space-ai-company-ubotica-unveils-breakthrough-research-in-dynamic-targeting-for-earth-observation
- ✓✓ **[chien2005eo1]** Chien, S., Sherwood, R., Tran, D., Cichy, B., Rabideau, G., Castaño, R.,
  Davies, A., Mandl, D., Frye, S., Trout, B., Shulman, S., Boyer, D. *Using Autonomy Flight Software
  to Improve Science Return on Earth Observing One.* Journal of Aerospace Computing, Information, and
  Communication, 2(4), 196–216, 2005. doi:10.2514/1.12923

## Onboard ML hardware

- ✓✓ **[giuffrida2022phisat]** Giuffrida, G., et al. *The Φ-Sat-1 Mission: The First On-Board Deep
  Neural Network Demonstrator for Satellite Earth Observation.* IEEE Transactions on Geoscience and
  Remote Sensing, 60, 1–14, 2022. doi:10.1109/TGRS.2021.3125567
  — Key facts used: CloudScout cloud-detection CNN on an Intel Movidius Myriad 2; launched
  3 Sept 2020.

## Missions and instruments

- ✓✓ **[illingworth2015earthcare]** Illingworth, A. J., et al. *The EarthCARE Satellite: The Next Step
  Forward in Global Measurements of Clouds, Aerosols, Precipitation, and Radiation.* Bulletin of the
  American Meteorological Society, 96(8), 1311–1332, 2015. doi:10.1175/BAMS-D-12-00227.1
- ✓✓ **[stephens2002cloudsat]** Stephens, G. L., Vane, D. G., Boain, R. J., et al. *The CloudSat
  Mission and the A-Train: A New Dimension of Space-Based Observations of Clouds and Precipitation.*
  Bulletin of the American Meteorological Society, 83(12), 1771–1790, 2002.
  doi:10.1175/BAMS-83-12-1771
- ✓✓ **[peral2019raincube]** Peral, E., et al. *RainCube: the first ever radar measurements from a
  CubeSat in space.* Journal of Applied Remote Sensing, 13(3), 032504, 2019.
  doi:10.1117/1.JRS.13.032504 — mission ended 24 Dec 2020.
- ✓✓ **[tanelli2023incus]** Tanelli, S., Haddad, Z. S., van den Heever, S. C., et al. *Investigation of
  Convective Updrafts (INCUS): status after phase A.* Proc. SPIE 12689, 1268904, 2023.
  doi:10.1117/12.2677734 — three smallsats, each with a RainCube-heritage Ka-band scanning radar;
  NASA lists launch in 2027. Mission page: https://incus.colostate.edu/mission/overview
- ✓ **[earthcare_status2025]** EarthCARE update, CCRES workshop, May 2025 (notes MSI calibration
  issues).
  https://www.actris.eu/sites/default/files/inline-files/20250520_EarthCARE_Update_CCRES_Work_shop.pptx_.pdf
  — workshop slides; prefer a peer-reviewed EarthCARE commissioning/validation paper if one exists by
  submission.
- ✓ **[nasa_trmm]** NASA GPM, *TRMM mission* and *TRMM satellite* pages.
  https://gpm.nasa.gov/missions/trmm · https://gpm.nasa.gov/missions/TRMM/satellite
  — Key facts used: NASA/JAXA, 1997–2015; PR (active), TMI and VIRS (passive) on one spacecraft;
  non-sun-synchronous, 35° inclination, ~46-day precession relative to the Sun; tropical/subtropical
  coverage; instruments differ in resolution and swath.
- ✓ **[nasa_gpm_core]** NASA GPM, *GPM Core Observatory* and *GPM mission* pages.
  https://gpm.nasa.gov/missions/GPM/core-observatory · https://gpm.nasa.gov/missions/GPM
  — Key facts used: NASA/JAXA, launched 2014; DPR (Ku/Ka) and GMI on one spacecraft;
  non-sun-synchronous, ~65° inclination; extends TRMM toward light rain and falling snow.
- ✓ **[nasa_gmi]** NASA GPM, *GPM Microwave Imager (GMI)* page. https://gpm.nasa.gov/missions/gpm/GMI
  — Key facts used: 13 channels; sensitive to rain, frozen precipitation, water vapor; GMI and DPR
  overlapping observations ~67 s apart.
- ? **[kummerow1998trmm]** Kummerow, C., Barnes, W., Kozu, T., Shiue, J., Simpson, J. *The Tropical
  Rainfall Measuring Mission (TRMM) Sensor Package.* Journal of Atmospheric and Oceanic Technology,
  15(3), 809–817, 1998. — details from recall; confirm volume, pages, and DOI before citing.
- ? **[hou2014gpm]** Hou, A. Y., Kakar, R. K., Neeck, S., et al. *The Global Precipitation
  Measurement Mission.* Bulletin of the American Meteorological Society, 95(5), 701–722, 2014.
  doi:10.1175/BAMS-D-13-00164.1 — details from recall; confirm before citing.

## Data and simulation

- ✓✓ **[gelaro2015g5nr]** Gelaro, R., Putman, W. M., Pawson, S., Draper, C., Molod, A., et al.
  *Evaluation of the 7-km GEOS-5 Nature Run.* NASA/TM-2014-104606, Vol. 36, Technical Report Series on
  Global Modeling and Data Assimilation, R. D. Koster (ed.), March 2015.
  https://ntrs.nasa.gov/citations/20150011486
  — Key facts used: two-year, 7 km, non-hydrostatic, built for OSSEs; realistic extremes and diurnal
  land precipitation; **known flaw: split ITCZ and a weaker-than-observed Hadley circulation.**
- ✓ **[g5nr_docs]** GMAO, *GEOS-5 Nature Run, Ganymed Release* documentation.
  https://gmao.gsfc.nasa.gov/global_mesoscale/7km-G5NR/docs
- ✓ **[lemoigne2017tatc]** Le Moigne, J., et al. *Tradespace Analysis Tool for Designing
  Constellations (TAT-C).* IGARSS 2017 — not located in IEEE Xplore yet; a related NTRS paper exists:
  https://ntrs.nasa.gov/citations/20180005607

## Methods (cite if used)

- ? Pedregosa, F., et al. *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830, 2011.
- ? Ke, G., et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
- ? A block-bootstrap reference for time-series resampling (e.g., Künsch 1989, Annals of Statistics).

## Own prior work and data

- ✓ Raj, S. GCSP report, Arizona State University, 2025.
  https://forge.engineering.asu.edu/wp-content/uploads/2025/06/Raj_GCSP.pdf
- Dataset DOI: [create on Zenodo before submission]
- Code DOI / repository tag: [create before submission]
