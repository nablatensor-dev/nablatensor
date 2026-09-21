package com.nablatensor.cva;
import com.nablatensor.codegen.Of;
@Of public final class CollateralAgreement {
  private final double threshold, minimumTransfer, independentAmount, marginPeriodOfRiskDays;
  private CollateralAgreement(double threshold,double minimumTransfer,double independentAmount,double marginPeriodOfRiskDays){if(!(threshold>=0&&minimumTransfer>=0&&independentAmount>=0&&marginPeriodOfRiskDays>=0))throw new IllegalArgumentException("all CSA terms must be >= 0");this.threshold=threshold;this.minimumTransfer=minimumTransfer;this.independentAmount=independentAmount;this.marginPeriodOfRiskDays=marginPeriodOfRiskDays;}
  static CollateralAgreement create(double threshold,double minimumTransfer,double independentAmount,double marginPeriodOfRiskDays){
    return new CollateralAgreement(threshold, minimumTransfer, independentAmount, marginPeriodOfRiskDays);
  }
  public static CollateralAgreementBuilder of(){return new CollateralAgreementBuilder();}
  public static CollateralAgreement uncollateralised(){return of().threshold(Double.POSITIVE_INFINITY).minimumTransfer(0).independentAmount(0).marginPeriodOfRiskDays(0).build();}
  public static CollateralAgreement dailyMargined(double amount){return of().threshold(0).minimumTransfer(0).independentAmount(amount).marginPeriodOfRiskDays(10).build();}
  public double threshold(){return threshold;} public double minimumTransfer(){return minimumTransfer;} public double independentAmount(){return independentAmount;} public double marginPeriodOfRiskDays(){return marginPeriodOfRiskDays;}
  public boolean isCollateralised(){return Double.isFinite(threshold);}
  public int marginPeriodSteps(double stepYears){return !isCollateralised()?0:Math.max(1,(int)Math.round(marginPeriodOfRiskDays/365.0/stepYears));}
}
