# QuaRAD
The Quadrature-based model of Respiratory Aerosol and Droplets (QuaRAD) is an efficient framework for simulating the full life cycle of virus-laden particles within indoor spaces --- from their initial creation and expulsion from an infectious person to their eventual removal from the space or their deposition in the nasal passages of a new host. The model considers interactions between an infectious person and a susceptible host. The model represents the release of virus-laden particles as either continuous emissions (e.g. speaking) or as an instantaneous puff (e.g. cough). The model output includes the virion concentration, the rate at which virions deposit to the nasal epithelium of a new host, and the risk of initial infection, all predicted as a function of the duration of the encounter and the location of the susceptbile person relative to the infectious person. 

Version 1.0.0

Released 2021-3-10

## References
Fierce, L., Robey, A. J., & Hamilton, C. (2021). Simulating near‐field enhancement in transmission of airborne viruses with a quadrature‐based model. Indoor air, 31(6), 1843-1859.


Fierce, L., Robey, A. J., & Hamilton, C. (2022). High efficacy of layered controls for reducing exposure to airborne pathogens. Indoor air, 32(2), e12989.

## Dependencies
NumPy, SciPy, matplotlib, pandas, SALib, pickle, os, pyDOE


## License
GNU General Public License v3.0
