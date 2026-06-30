# Getting `MANO_RIGHT.pkl`

The WiLoR method (`method="wilor"`) needs the MANO right-hand model,
`MANO_RIGHT.pkl`. It is not redistributable - you download it yourself after
agreeing to the MANO license, which permits non-commercial research use only.

## Steps

1. Go to the [MANO website](https://mano.is.tue.mpg.de) and click **Sign Up** to
   create an account (email verification required), then sign in.
2. Open the **Download** page and accept the
   [MANO license](https://mano.is.tue.mpg.de/license.html).
3. Download the models archive (`mano_v*_*.zip`, currently `mano_v1_2.zip`).
4. Unzip it. The right-hand model is at `mano_v1_2/models/MANO_RIGHT.pkl`.

## Using it

Point `track_hands` at the file you extracted:

```python
df = df.with_column(
    "hands",
    track_hands(df["observation.image"], method="wilor", mano_path="/path/to/MANO_RIGHT.pkl"),
)
```

Or with the CLI: `daft-physical-ai --method wilor --mano-path /path/to/MANO_RIGHT.pkl ...`.

You only need to supply the single `MANO_RIGHT.pkl` file - the package's
`ensure_assets` fetches the rest of the WiLoR repo and weights and places the
MANO file where WiLoR expects it.
