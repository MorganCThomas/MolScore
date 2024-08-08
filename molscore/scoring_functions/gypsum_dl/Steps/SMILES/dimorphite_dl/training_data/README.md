Training Data
=============

Format
------

To allow others to reproduce our work, we here include the data used to
calculate typical pKa ranges for 38 ionizable substructures. Please see the
`training_data.json` file.

The keys of the JSON are the labels of each substructure (e.g.,
"Thioic_acid"). The JSON values are lists of pKa values. For example:

``` json
{
    "Aromatic_protonated_nitrogen": [
        7.7, 14.9, 15.3, ...
    ],
    "Vynl_alcohol": [
        9.2, 9.5, 9.5, ...
    ]
}
```

In the case of "Phosphate" and "Phosphonate" groups, the values are lists of
two pKa values (pKa1 and pKa2). Where one of these pKa values is unavailable,
it is listed as `null`. For example:

``` json
{
    "Phosphonate": [
        [1.1, 6.5], [2.7, 8.4], [null, 8.7], ...
    ]
}
```

Reaxys Terms and Conditions
---------------------------

Most of the pKa data used to train Dimorphite-DL was taken from the [Reaxys
database](https://www.reaxys.com/#/about-content), owned and operated by
Elsevier Information Systems GmbH. [Facts are not
copyrightable](https://www.copyright.gov/help/faq/faq-protect.html), but in
using the database we did agree to Elsevier's [Terms and
Conditions](https://www.elsevier.com/legal/elsevier-website-terms-and-conditions).

Ideally, we would like to include both the SMILES strings and precise
catalogued pKa values for all training examples. But, given the Terms and
Conditions, it is unclear whether this use is permissible:

> Unless otherwise set out herein, content comprised within the Services,
> including text... and other information (collectively, the "Content")... is
> owned by Elsevier, its licensors or its content providers and is protected
> by copyright, trademark and other intellectual property and unfair
> competition laws.

Do the catalogued SMILES strings and pKa values fall under this definition of
"content"? But they are not copyrightable, perhaps suggesting they do not. On
the other hand, publication is certainly a kind of "scholarly use":

> ...you may print or download Content from the Services for your own
> personal, non-commercial, informational or scholarly use, provided that you
> keep intact all copyright and other proprietary notices.

But, later in the terms, publication seems to be expressly prohibited:

> You may not copy, display, distribute, modify, publish, reproduce, store,
> transmit, post, translate or create other derivative works from, or sell,
> rent or license all or any part of the Content... in any medium to anyone,
> except as otherwise expressly permitted under these Terms and Conditions, or
> any relevant license or subscription agreement or authorization by us.

We emailed Reaxys seeking clarification but did not hear back from them.

Solution
--------

Given this uncertainty, we opted not to publish the exact SMILES structures
taken from the Reaxys database. We further opted to round the pKa values to
the nearest tenth, to avoid directly redistributing Reaxys data. The data we
do provide should allow others to recalculate our pKa ranges with reasonable
accuracy.
