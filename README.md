# BERTAVP: an interpretable multi-task learning model for identification and functional prediction of antiviral peptides
Antiviral peptides (AVPs) have potential to enhance our response to viral diseases, making them a valuable addition to therapeutic options. Most studies do not analyze the focus of their models or lack predictions for subclasses. Therefore, it is necessary to have a functional prediction of AVPs with an interpretable model. However, developing predictive and interpretable models remains a challenge because of restricted peptides representation and data imbalance. Here, we introduce BERTAVP, an interpretable deep learning framework, for identifying AVPs and characterizing their functional activities (eight species and six families). First, it utilizes the BERT branch to extract peptide features and CNN branch to extract amino acid and physicochemical features. It then applies the focal loss to mitigate the negative impact caused by imbalanced datasets. Finally, we further analyze biological patterns and key motifs learned by BERTAVP, finding ELDKWA and SLWNWF motifs demonstrate the strongest antiviral properties within the AVPs. Experimental results on public datasets show our model achieves superior performance in identification and functional prediction of AVPs.
![model](./images/figure2.png)
