package com.nablatensor.cva;
import com.nablatensor.codegen.Of;
import com.nablatensor.risk.RiskClassEnum;
import com.nablatensor.risk.RiskFactor;
import java.util.Map;
@Of public final class SaCvaParameters {
  private final double mCva,creditSpreadVegaRw,girrDeltaRw,girrVegaRw,fxDeltaRw,fxVegaRw,creditSpreadRho,creditSpreadGamma,fxGamma;
  private final Map<String,Double> creditSpreadRw;
  private SaCvaParameters(double mCva,double creditSpreadVegaRw,double girrDeltaRw,double girrVegaRw,double fxDeltaRw,double fxVegaRw,double creditSpreadRho,double creditSpreadGamma,double fxGamma,Map<String,Double> creditSpreadRw){if(!(mCva>=1))throw new IllegalArgumentException("m_CVA has a supervisory floor of 1.0, got "+mCva);this.mCva=mCva;this.creditSpreadRw=Map.copyOf(creditSpreadRw);this.creditSpreadVegaRw=creditSpreadVegaRw;this.girrDeltaRw=girrDeltaRw;this.girrVegaRw=girrVegaRw;this.fxDeltaRw=fxDeltaRw;this.fxVegaRw=fxVegaRw;this.creditSpreadRho=creditSpreadRho;this.creditSpreadGamma=creditSpreadGamma;this.fxGamma=fxGamma;}
  static SaCvaParameters create(double mCva,double creditSpreadVegaRw,double girrDeltaRw,double girrVegaRw,double fxDeltaRw,double fxVegaRw,double creditSpreadRho,double creditSpreadGamma,double fxGamma,Map<String,Double> creditSpreadRw){
    return new SaCvaParameters(mCva, creditSpreadVegaRw, girrDeltaRw, girrVegaRw, fxDeltaRw, fxVegaRw, creditSpreadRho, creditSpreadGamma, fxGamma, creditSpreadRw);
  }
  public static SaCvaParametersBuilder of(){return new SaCvaParametersBuilder();}
  public static SaCvaParameters demo(){return of().mCva(1).creditSpreadRw(Map.of("1",.005,"2",.010,"3",.050,"4",.030,"5",.030,"6",.020,"7",.120)).creditSpreadVegaRw(1).girrDeltaRw(.0111).girrVegaRw(1).fxDeltaRw(.11).fxVegaRw(1).creditSpreadRho(.9).creditSpreadGamma(.5).fxGamma(.6).build();}
  public double mCva(){return mCva;} public Map<String,Double> creditSpreadRw(){return creditSpreadRw;} public double creditSpreadVegaRw(){return creditSpreadVegaRw;} public double girrDeltaRw(){return girrDeltaRw;} public double girrVegaRw(){return girrVegaRw;} public double fxDeltaRw(){return fxDeltaRw;} public double fxVegaRw(){return fxVegaRw;} public double creditSpreadRho(){return creditSpreadRho;} public double creditSpreadGamma(){return creditSpreadGamma;} public double fxGamma(){return fxGamma;}
  public double deltaRiskWeight(RiskFactor f){return switch(f.riskClass()){case GIRR->girrDeltaRw;case CSR_NON_SEC->creditSpreadRw.getOrDefault(f.bucket(),.05);case FX->fxDeltaRw;default->throw new IllegalArgumentException("no SA-CVA delta RW for "+f.riskClass());};}
  public double vegaRiskWeight(RiskFactor f){return switch(f.riskClass()){case GIRR->girrVegaRw;case CSR_NON_SEC->creditSpreadVegaRw;case FX->fxVegaRw;default->throw new IllegalArgumentException("no SA-CVA vega RW for "+f.riskClass());};}
  public double withinBucketCorrelation(RiskClassEnum c){return c==RiskClassEnum.CSR_NON_SEC?creditSpreadRho:.99;}
  public double acrossBucketCorrelation(RiskClassEnum c){return switch(c){case CSR_NON_SEC->creditSpreadGamma;case FX->fxGamma;default->0;};}
}
