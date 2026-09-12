import equinox as eqx
class PKPDParams(eqx.Module):
    """Fixed biophysical parameters for semaglutide PK/PD."""

    k_a: Float[Array, ""]
    k_e: Float[Array, ""]
    v_d: Float[Array, ""]
    bioavailability: Float[Array, ""]
    e_max: Float[Array, ""]
    e_c50: Float[Array, ""]
    hill: Float[Array, ""]

    @classmethod
    def semaglutide(cls) -> PKPDParams:
        """Literature-scale defaults for once-weekly SC semaglutide."""
        t_half = 7.0
        return cls(
            k_a=jnp.asarray(0.75), # absorption rate constant
            k_e=jnp.asarray(jnp.log(2.0) / t_half), # elimination rate constant
            v_d=jnp.asarray(12.5), # volume of distribution
            bioavailability=jnp.asarray(0.89), # fraction of dose absorbed
            e_max=jnp.asarray(1.0), # maximum effect
            e_c50=jnp.asarray(0.05), # concentration at which 50% of the maximum effect is achieved
            hill=jnp.asarray(1.0), # hill coefficient
        )

